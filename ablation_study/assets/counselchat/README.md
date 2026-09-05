---
language:
- en
license: openrail
source:
- https://huggingface.co/datasets/loaiabdalslam/counselchat
dataset_info:
  features:
  - name: questionID
    dtype: string
  - name: questionTitle
    dtype: string
  - name: questionText
    dtype: string
  - name: questionUrl
    dtype: string
  - name: topics
    dtype: string
  - name: therapistName
    dtype: string
  - name: therapistUrl
    dtype: string
  - name: answerText
    dtype: string
  - name: upvotes
    dtype: int64
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 5791592
    num_examples: 1482
  download_size: 3346530
  dataset_size: 5791592
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# CounselChat snapshot

This is the frozen 1,482-row snapshot used by the experiment. It was obtained from [`loaiabdalslam/counselchat`](https://huggingface.co/datasets/loaiabdalslam/counselchat), which labels the dataset with the OpenRAIL license and derives from the public CounselChat corpus.

The records contain mental-health-related text and named therapist metadata. Although the original question authors are presented as anonymous, the content may still be sensitive. Use the snapshot for research with appropriate safeguards, follow the source dataset's license and terms, and do not treat it as clinical advice.

The experiment reads only `questionID`, `questionTitle`, `questionText`, and `topics`; therapist answers and profile metadata are not used by the agent pipeline.
