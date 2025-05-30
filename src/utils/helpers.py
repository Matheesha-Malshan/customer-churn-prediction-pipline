from sklearn.model_selection import train_test_split
from prefect import task

@task
def split_data(x, y):
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    return x_train, x_val, x_test, y_train, y_val, y_test


