import os, argparse, time, sqlite3, pickle
from tqdm import tqdm
import torch
from transformers import T5ForConditionalGeneration
from load_data import get_dataloader

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DB_PATH = 'data/flight_database.db'


def execute_query(query):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        rec = cursor.fetchall()
        error_msg = ""
    except Exception as e:
        rec = []
        error_msg = f"{type(e).__name__}: {e}"
    conn.close()
    return rec, error_msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--experiment_name', type=str, default='experiment')
    parser.add_argument('--test_batch_size', type=int, default=16)
    args = parser.parse_args()

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name, 'best')

    print(f"Loading model from {checkpoint_dir}")
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)
    model.to(DEVICE)
    model.eval()

    test_loader = get_dataloader(args.test_batch_size, "test")
    tokenizer = test_loader.dataset.tokenizer

    # Generate SQL
    generated_sql = []
    print("Generating SQL...")
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                do_sample=False,
                max_new_tokens=512,
            )
            generated_sql.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    sql_path = f'results/t5_{model_type}_{args.experiment_name}_test.sql'
    with open(sql_path, 'w') as f:
        for q in generated_sql:
            f.write(f'{q}\n')

    print("Computing records...")
    records = []
    error_msgs = []
    for query in tqdm(generated_sql):
        rec, err = execute_query(query)
        records.append(rec)
        error_msgs.append(err)

    record_path = f'records/t5_{model_type}_{args.experiment_name}_test.pkl'
    with open(record_path, 'wb') as f:
        pickle.dump((records, error_msgs), f)

    print(f"Saved:\n  {sql_path}\n  {record_path}")


if __name__ == "__main__":
    main()
