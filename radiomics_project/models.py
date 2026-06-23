
# 7. 模型训练与预测


def get_model(model_name):
    if model_name == "LogisticRegression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=5000,
            )),
        ])

    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])

    if model_name == "XGBoost":
        if not HAS_XGB:
            raise RuntimeError(f"当前环境没有 xgboost。导入失败原因：{XGB_IMPORT_ERROR}。请确认使用安装了 xgboost 的 Python 解释器。")
        return XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    raise ValueError(f"未知模型：{model_name}")


def fit_model(model_name, x_train, y_train, labels_sorted):
    model = get_model(model_name)

    if model_name == "XGBoost":
        label_to_num = {labels_sorted[0]: 0, labels_sorted[1]: 1}
        y_train_num = np.array([label_to_num[v] for v in y_train])
        model.fit(x_train, y_train_num)
    else:
        model.fit(x_train, y_train)

    return model


def predict_with_model(model, model_name, x_test, labels_sorted, positive_label):
    if model_name == "XGBoost":
        pred_num = model.predict(x_test)
        prob = model.predict_proba(x_test)[:, 1]
        num_to_label = {0: labels_sorted[0], 1: labels_sorted[1]}
        pred = np.array([num_to_label[int(v)] for v in pred_num])
        return pred, prob

    pred = model.predict(x_test)

    if isinstance(model, Pipeline):
        class_list = list(model.named_steps["clf"].classes_)
    else:
        class_list = list(model.classes_)

    pos_index = class_list.index(positive_label)
    prob = model.predict_proba(x_test)[:, pos_index]
    return pred, prob


def fit_predict_model(model_name, x_train, y_train, x_test, labels_sorted, positive_label):
    model = fit_model(model_name, x_train, y_train, labels_sorted)
    pred, prob = predict_with_model(model, model_name, x_test, labels_sorted, positive_label)
    return pred, prob, model


