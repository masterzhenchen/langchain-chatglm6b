from typing import Optional, List
import torch
from rich import print


def mlm_loss(logits, mask_positions, sub_mask_labels, cross_entropy_criterion, device):
    batch_size, seq_len, vocab_size = logits.size()
    loss = None
    for single_value in zip(logits, sub_mask_labels, mask_positions):
        single_logits = single_value[0]
        single_sub_mask_labels = single_value[1]
        single_mask_positions = single_value[2]

        # single_mask_logits形状：(mask_label_num, vocab_size)
        single_mask_logits = single_logits[single_mask_positions]
        single_mask_logits = single_mask_logits.repeat(len(single_sub_mask_labels), 1, 1)
        single_mask_logits = single_mask_logits.reshape(-1, vocab_size)
        # single_sub_mask_label 形状:(sub_label_num,mask_label_num)
        single_sub_mask_labels = torch.LongTensor(single_sub_mask_labels).to(device)
        single_sub_mask_labels = single_sub_mask_labels.reshape(-1, 1).squeeze()
        cur_loss = cross_entropy_criterion(single_mask_logits, single_sub_mask_labels)
        cur_loss = cur_loss / len(single_sub_mask_labels)
        if not loss:
            loss = cur_loss
        else:
            loss += cur_loss
    loss = loss / batch_size
    return loss


def convert_logits_to_ids(
        logits: torch.tensor,
        mask_positions: torch.tensor):
    """
    输入LM的词表概率分布（LMModel的logits），将mask_position位置的
    token logits转换为token的id。

    Args:
        logits (torch.tensor): model output -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token的位置 -> (batch, mask_label_num)

    Returns:
        torch.LongTensor: 对应mask position上最大概率的推理token -> (batch, mask_label_num)
    """
    label_length = mask_positions.size()[1]  # 标签长度
    # print(f'label_length--》{label_length}')
    batch_size, seq_len, vocab_size = logits.size()

    mask_positions_after_reshaped = []

    for batch, mask_pos in enumerate(mask_positions.detach().cpu().numpy().tolist()):
        for pos in mask_pos:
            mask_positions_after_reshaped.append(batch * seq_len + pos)

    # logits形状：(batch_size * seq_len, vocab_size)
    logits = logits.reshape(batch_size * seq_len, -1)

    # mask_logits形状：(batch * label_num, vocab_size)
    mask_logits = logits[mask_positions_after_reshaped]

    # predict_tokens形状： (batch * label_num)
    predict_tokens = mask_logits.argmax(dim=-1)

    # 改变后的predict_tokens形状： (batch, label_num)
    predict_tokens = predict_tokens.reshape(-1, label_length)  # (batch, label_num)

    return predict_tokens
