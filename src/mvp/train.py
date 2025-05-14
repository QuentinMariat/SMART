import transformers
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from src.data.data_handler import load_and_preprocess_data
from src.models.hf_model import get_model
from src.evaluation.metrics import compute_metrics
from src.config.settings import TRAINING_ARGS, ID2LABEL, LABEL2ID, NUM_LABELS

def train_model():
    """
    charge les données et le modèle, puis lance le fine tunning.
    """
    print("Starting training process...")

    train_dataset, val_dataset, test_dataset, tokenizer = load_and_preprocess_data()

    # charger le modèle
    model = get_model()

    # configurer les arguments d'entraînement
    training_args = TrainingArguments(
        output_dir=TRAINING_ARGS["output_dir"],
        num_train_epochs=TRAINING_ARGS["num_train_epochs"],
        per_device_train_batch_size=TRAINING_ARGS["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_ARGS["per_device_eval_batch_size"],
        warmup_steps=TRAINING_ARGS["warmup_steps"],
        weight_decay=TRAINING_ARGS["weight_decay"],
        logging_dir=TRAINING_ARGS["logging_dir"],
        logging_steps=TRAINING_ARGS["logging_steps"],
        evaluation_strategy=TRAINING_ARGS["evaluation_strategy"],
        save_strategy=TRAINING_ARGS["save_strategy"],
        load_best_model_at_end=TRAINING_ARGS["load_best_model_at_end"],
        metric_for_best_model=TRAINING_ARGS["metric_for_best_model"],
        greater_is_better=TRAINING_ARGS["greater_is_better"],
    )

    # configurer du data collator : responsable du padding des séquences dans chaque batch.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # initialiser le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # lancer l'entraînement
    print("Training started...")
    trainer.train()
    print("Training finished.")

    # sauvegarder le modèle final
    # le Trainer sauvegarde déjà automatiquement le meilleur modèle dans le répertoire output_dir spécifié. Vous pouvez spécifier un sous-dossier si vous voulez.
    final_model_path = f"{TRAINING_ARGS['output_dir']}/mvp_model"
    print(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    print("Final model saved.")

if __name__ == "__main__":
    train_model()