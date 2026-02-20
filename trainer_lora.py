import pandas as pd
import torch
from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from transformers import Adafactor


def main():
    torch.cuda.empty_cache()


    df_train = pd.read_csv("ОбучающийТриплет.csv", sep="|", encoding="utf-8")
    df_test  = pd.read_csv("ТестовыйТриплет.csv",  sep="|", encoding="utf-8")

    train_dataset = Dataset.from_dict({
        "anchor":   df_train["anchor"].astype(str).tolist(),
        "positive": df_train["positive"].astype(str).tolist(),
        "negative": df_train["negative"].astype(str).tolist(),
    })


    model = SentenceTransformer("ai-forever/FRIDA")

    model[0].auto_model.gradient_checkpointing_enable()

    loss = MultipleNegativesRankingLoss(model)

    test_queries, test_corpus, test_relevant = {}, {}, {}
    for idx, row in df_test.iterrows():
        qid    = f"q{idx}"
        doc_id = f"doc{idx}"
        test_queries[qid]  = str(row["anchor"])
        test_corpus[doc_id] = str(row["positive"])
        test_relevant[qid]  = {doc_id}

    evaluator = InformationRetrievalEvaluator(
        test_queries, test_corpus, test_relevant, name="rag_eval"
    )

    args = SentenceTransformerTrainingArguments(
        output_dir="./frida-full-ft",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,

        bf16=True,
        fp16=False,

        optim="adafactor",
        optim_args="scale_parameter=False,relative_step=False,warmup_init=False",
        learning_rate=3e-5,

        max_grad_norm=1.0,
        warmup_ratio=0.1,
        weight_decay=0.01,

        logging_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,

        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()
    model.save_pretrained("./frida-full-ft-final")


if __name__ == "__main__":
    main()
