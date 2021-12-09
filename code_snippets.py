# Additional code snippet used for finding optimal threshold value.

threshold =[i * 0.1 for i in range(1, 19)]

for i in threshold:

    forest_pipe = Pipeline(
        steps=[
            ("selector", SelectFromModel(estimator=forest, threshold=f"{i}*mean")),
            ("model", forest)
        ]
    )

    forest_pipe.fit(X_train, y_train)
    predictions = forest_pipe.predict(X_test)

    output = pd.DataFrame({"id": test["id"], "target": predictions})
    output.to_csv(f"submissions/temp/forest_{i}_mean.csv", index=False)