import torch


def extract_embeddings(batch, model):
    input_ids = [torch.tensor(seq) for seq in batch['input_ids']]
    attention_mask = [torch.tensor(seq) for seq in batch['attention_mask']]

    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Mean Pooling
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return {'embeddings': embeddings}
