// Plain-English definitions for every technical term the dashboard uses.
// Shown on hover (Term component) and listed in the Guide.

export const GLOSSARY = {
  accuracy: {
    term: "Accuracy",
    definition: "The share of invoice lines the model labels correctly.",
  },
  baseline: {
    term: "Baseline",
    definition:
      "The score of the simplest possible rule: always answering the most common category. A useful model must beat it.",
  },
  holdout: {
    term: "Holdout",
    definition:
      "10% of the data (590 lines) set aside before any work started and used once, at the very end, to check the finished model on data it had never seen.",
  },
  "cross-validation": {
    term: "Cross-validation",
    definition:
      "Testing a model by splitting the training data into 5 parts, training on 4 and testing on the 5th, in turn. Repeated with 3 random seeds, so every score averages 15 tests.",
  },
  "grouped cross-validation": {
    term: "Grouped cross-validation",
    definition:
      "Cross-validation where exact copies of a row always land in the same part. Without it, the model would be tested on rows it had already seen in training, and the score would look better than it is.",
  },
  "out-of-time": {
    term: "Out-of-time test",
    definition:
      "Train on the months up to 2018-09 and test on the months after, the way the model would be used on future invoices.",
  },
  "macro F1": {
    term: "Macro F1",
    definition:
      "The F1 score (a balance of precision and recall) computed for each category and then averaged, so a rare category counts as much as a common one. It covers the five categories with 11 or more training rows.",
  },
  precision: {
    term: "Precision",
    definition: "Of the lines the model put in a category, the share that really belong there.",
  },
  recall: {
    term: "Recall",
    definition: "Of the lines that really belong to a category, the share the model finds.",
  },
  confidence: {
    term: "Confidence",
    definition:
      "The model's probability for its suggested category. After calibration, lines given 90% confidence are right about 9 times in 10.",
  },
  calibration: {
    term: "Calibration",
    definition:
      "Adjusting the model's confidence so it is honest: one number (the temperature) rescales every score so that stated confidence matches observed accuracy.",
  },
  "review threshold": {
    term: "Review threshold",
    definition:
      "The confidence below which a line is sent to a person. Lines with a critical warning (such as a new vendor) are sent too, whatever their confidence.",
  },
  "log-odds": {
    term: "Log-odds",
    definition:
      "The model's internal score for one category against another. Contributions in log-odds add up exactly: positive values push toward the suggested category, negative values toward the runner-up.",
  },
  "base rate": {
    term: "Base rate",
    definition:
      "The part of the score that comes from how common each category is, before looking at the line's own details.",
  },
  blend: {
    term: "Log-odds blend",
    definition:
      "The final model: a weighted sum of the scores of a logistic regression and a LightGBM model. The two make different mistakes, so the blend beats each one.",
  },
  "TF-IDF": {
    term: "TF-IDF",
    definition:
      "A way to turn words into numbers: each word is weighted by how often it appears in a line and how rare it is overall, so distinctive words count more.",
  },
  ablation: {
    term: "Ablation",
    definition: "Removing one group of inputs, retraining, and measuring how much the score changes.",
  },
  "Cramer's V": {
    term: "Cramer's V",
    definition: "A 0-to-1 measure of how strongly two categorical variables are linked; 1 means one decides the other.",
  },
  "mutual information": {
    term: "Mutual information",
    definition: "How much knowing an input reduces uncertainty about the category (in nats; 0 = nothing).",
  },
  "class weights": {
    term: "Class weights",
    definition:
      "Telling the model to pay more attention to rare categories during training, so it does not simply ignore them.",
  },
} as const

export type GlossaryKey = keyof typeof GLOSSARY
