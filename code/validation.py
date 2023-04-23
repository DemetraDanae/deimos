# q2 external calculation

def q2ext(y_test, y_test_predicted, y_train):

    import statistics as st

    average_train = st.mean(y_train)

    q2 = 1- sum((y_test.values-y_test_predicted.squeeze())**2)/sum((y_test.values-average_train)**2)

    return q2
