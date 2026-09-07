# Experiments

This directory groups all experimental materials in the workspace.

- [original_studies/](original_studies/README.md): CPED emotion recognition, independent Need/Topic classification, and the original local-model Agent comparison, including predictions, statistical tests, figures, and prompt-development records.

- `ablation_study/`: executable agent, frozen CounselChat data, the six-condition ablation protocol, raw/processed outputs, figures and the final report.
- `user_study/`: scripts used to prepare the English survey export and the de-identified public dataset.

Each experiment has its own README with inputs, procedure, outputs and privacy notes.

## Datasets and Sources

- **CPED (Chinese Personalized and Emotional Dialogue Dataset)**  
  Source: https://github.com/scutcyr/CPED  
  Used for contextual emotion recognition experiments.

- **Mental Health Counseling Conversations**  
  Source: https://www.kaggle.com/datasets/melissamonfared/mental-health-counseling-conversations-k  
  Used for need/topic classification experiments.

- **CounselChat**  
  Source: https://huggingface.co/datasets/loaiabdalslam/counselchat  
  Used in the supplementary reconstructed ablation study.

- **User Study Dataset**  
  Collected through the questionnaire conducted for this thesis.  
  Only de-identified records are included in the public release.
