import csv

import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from api.model import predict


@pytest.mark.parametrize("payload", [
    "data/ml_test_dataset.csv", "data/ml_test_dataset_1000.csv"
])
def test_model_evaluation(payload):
    input_file = payload
    output_file = "results.csv"

    texts = []
    expected_labels = []
    predicted_labels = []
    confidences = []

    with open(input_file, newline='', encoding='utf-8') as infile, \
            open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = ['text', 'expected', 'predicted', 'confidence', 'is_correct']

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            text = row['text']
            expected = row['expected_label'].upper().strip()

            try:
                result = predict(text)[0]
                predicted = result['label'].upper()
                confidence = float(result['score'])

                is_correct = predicted == expected

                writer.writerow({
                    'text': text,
                    'expected': expected,
                    'predicted': predicted,
                    'confidence': round(confidence, 4),
                    'is_correct': is_correct
                })
                texts.append(text)
                expected_labels.append(expected)
                predicted_labels.append(predicted)
                confidences.append(confidence)
            except Exception as e:
                print(f"Error processing: {text} | Error: {e}")

        accuracy = accuracy_score(expected_labels, predicted_labels)
        precision = precision_score(expected_labels, predicted_labels, pos_label="POSITIVE", average='binary')
        recall = recall_score(expected_labels, predicted_labels, pos_label="POSITIVE", average='binary')
        f1 = f1_score(expected_labels, predicted_labels, pos_label="POSITIVE", average='binary')
        conf_matrix = confusion_matrix(expected_labels, predicted_labels)
        avg_confidence = sum(confidences) / len(confidences)

        print("\n--- Evaluation Report ---")
        print(f"Accuracy       : {accuracy:.4f}")
        print(f"Precision      : {precision:.4f}")
        print(f"Recall         : {recall:.4f}")
        print(f"F1 Score       : {f1:.4f}")
        print(f"Avg Confidence : {avg_confidence:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
