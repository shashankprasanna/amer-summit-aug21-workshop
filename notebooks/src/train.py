import os
import sys
import logging
import argparse
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer


def main(args):
        
    # Hyper-parameters
    training_dir     = args.training_dir
    test_dir         = args.test_dir
    output_dir       = args.output_dir
    output_data_dir  = args.output_data_dir
    model_dir        = args.model_dir

    model_name       = args.model_name
    epochs           = args.epochs
    train_batch_size = args.train_batch_size
    eval_batch_size  = args.eval_batch_size
    warmup_steps     = args.warmup_steps
    learning_rate    = args.learning_rate
    
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.info(sys.argv)
    
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
   
    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    train_file    = f"{training_dir}/train.csv"
    validate_file = f"{test_dir}/validate.csv"
    
    dataset = load_dataset('csv', data_files={'train': train_file,
                                             'test': validate_file})
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)
    
    # set format for pytorch
    train_dataset =  train_dataset.rename_column("label", "labels")
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

    # define training args
    training_args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = train_batch_size,
        per_device_eval_batch_size  = eval_batch_size,
        warmup_steps                = warmup_steps,
        evaluation_strategy         = "epoch",
        logging_dir                 = f"{output_data_dir}/logs",
        learning_rate               = float(learning_rate),
    )

    # create Trainer instance
    trainer = Trainer(
        model           = model,
        args            = training_args,
        compute_metrics = compute_metrics,
        train_dataset   = train_dataset,
        eval_dataset    = test_dataset,
    )

    # train model
    if get_last_checkpoint(output_dir) is not None:
        logger.info("***** continue training *****")
        trainer.train(resume_from_checkpoint=output_dir)
    else:
        trainer.train()
    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--output_dir", type=str,  default='/opt/ml/checkpoints')

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()
    
    main(args)
