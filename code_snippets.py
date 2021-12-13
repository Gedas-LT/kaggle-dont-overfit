# Additional code snippet used for finding optimal threshold value.

threshold =[i * 0.1 for i in range(1, 31)]

for i in threshold:

    log_pipe = Pipeline(
        steps=[
            ("selector", SelectFromModel(estimator=log_reg, threshold=f"{i}*mean")),
            ("model", log_reg)
        ]
    )

    log_pipe.fit(X_train, y_train)
    predictions = log_pipe.predict_proba(X_test)

    output = pd.DataFrame({"id": test["id"], "target": predictions[:, 1]})
    output.to_csv(f"submissions/temp/forest_{i}_mean.csv", index=False)