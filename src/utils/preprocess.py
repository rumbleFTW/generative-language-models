import re


def clean_text(raw_text: str):
    """
    Cleans `raw_text`. This can be modified to include/exclude punctuations, special characters and uppercase letters.
    """
    raw_text = re.sub(r"[^a-zA-Z ]", " ", raw_text.replace("\n", " ").lower())
    return re.sub(r"\s{2,}", " ", raw_text)


def split(X_data, y_data, ratio: float):
    """
    Split data into training and testing set with ratio `ratio`.
    """
    split_point = int(len(X_data) * ratio)

    X_train = X_data[:split_point]
    y_train = y_data[:split_point]

    X_val = X_data[split_point:]
    y_val = y_data[split_point:]

    return X_train, y_train, X_val, y_val
