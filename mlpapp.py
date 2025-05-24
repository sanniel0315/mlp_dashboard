import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- 數據加載與預處理 (使用 Streamlit 緩存) ---
@st.cache_data
def load_and_preprocess_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=iris.feature_names) # 轉回 DataFrame 以保留列名

    # 切分訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, iris.target_names, X.columns.tolist(), scaler # 將 feature_names 轉為列表

# 在應用程式啟動時加載和預處理數據
X_train_full, X_test_full, y_train, y_test, target_names, all_feature_names, loaded_scaler = load_and_preprocess_data()

st.title('MLP 模型訓練與超參數互動式探索')
st.markdown('透過調整左側邊欄的參數來訓練 MLP 模型並觀察結果。')

# --- 側邊欄參數設定 ---
st.sidebar.header('MLP 模型超參數設定')

# 新增的特徵選擇功能
st.sidebar.subheader('0. 特徵選擇')
selected_features = st.sidebar.multiselect(
    '選擇要包含的特徵',
    options=all_feature_names,
    default=all_feature_names # 預設全選
)

if not selected_features:
    st.sidebar.warning("請至少選擇一個特徵！")
    st.stop() # 如果沒有選擇任何特徵，則停止應用程式運行

# 根據選擇的特徵篩選數據集
X_train = X_train_full[selected_features]
X_test = X_test_full[selected_features]

# 1. 模型複雜度 (Hidden Layer Sizes)
st.sidebar.subheader('1. 模型複雜度')
hidden_layer_1 = st.sidebar.slider('第一隱藏層神經元數量', 10, 200, 100, step=10)
num_hidden_layers = st.sidebar.radio('隱藏層數量', [1, 2], index=0)

hidden_layer_sizes = (hidden_layer_1,)
if num_hidden_layers == 2:
    hidden_layer_2 = st.sidebar.slider('第二隱藏層神經元數量', 10, 100, 50, step=10)
    hidden_layer_sizes = (hidden_layer_1, hidden_layer_2)

st.sidebar.write(f'設定的隱藏層大小: {hidden_layer_sizes}')

# 2. 活化函數 (Activation Function)
activation_function = st.sidebar.selectbox(
    '2. 活化函數 (Activation Function)',
    ['relu', 'logistic', 'tanh', 'identity'],
    index=0 # 預設 ReLU
)

# 3. 優化器 / 求解器 (Solver)
solver = st.sidebar.selectbox(
    '3. 優化器 / 求解器 (Solver)',
    ['adam', 'sgd', 'lbfgs'],
    index=0 # 預設 Adam
)

# 4. 學習率 (Learning Rate) - 僅適用於 Adam 和 SGD
learning_rate_init = 0.001 # 預設初始學習率
if solver in ['adam', 'sgd']:
    st.sidebar.subheader('4. 學習率')
    learning_rate = st.sidebar.selectbox(
        '學習率策略',
        ['constant', 'invscaling', 'adaptive'],
        index=0 # 預設 constant
    )
    learning_rate_init = st.sidebar.number_input('初始學習率', min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f")
else:
    st.sidebar.info('L-BFGS 優化器不使用學習率相關參數。')
    learning_rate = 'constant' # 對 L-BFGS 無效，但為 API 參數完整性保留

# 5. 批次大小 (Batch Size) - 僅適用於 Adam 和 SGD
batch_size = 'auto' # 預設值
if solver in ['adam', 'sgd']:
    st.sidebar.subheader('5. 批次大小')
    batch_size_option = st.sidebar.radio('批次大小設定', ['auto', '手動輸入'], index=0)
    if batch_size_option == '手動輸入':
        batch_size = st.sidebar.number_input('Batch Size', min_value=1, max_value=len(X_train), value=32, step=1)
    else:
        st.sidebar.info('`auto` 會使用 `min(200, n_samples)`。')
else:
    st.sidebar.info('L-BFGS 優化器不使用批次大小。')
    batch_size = 'auto' # 對 L-BFGS 無效，但為 API 參數完整性保留

# 6. 訓練次數 / Epochs (Max Iter)
st.sidebar.subheader('6. 訓練次數')
max_iter = st.sidebar.slider('最大迭代次數 (Max Iter)', 50, 1000, 200, step=50)

# 7. Early Stopping (提早停止)
st.sidebar.subheader('7. 提早停止 (Early Stopping)')
early_stopping = st.sidebar.checkbox('啟用 Early Stopping', value=False)
if early_stopping:
    validation_fraction = st.sidebar.slider('驗證集比例 (Validation Fraction)', 0.05, 0.5, 0.1, step=0.05)
    n_iter_no_change = st.sidebar.slider('無改善容忍迭代次數', 10, 100, 10, step=5)
    tol = st.sidebar.number_input('容忍度 (Tolerance)', min_value=1e-5, max_value=1e-2, value=1e-4, format="%.5f")
else:
    st.sidebar.info('若不啟用 Early Stopping，模型會跑完所有 Max Iter。')
    n_iter_no_change = max_iter + 1
    tol = 1e-4

# 8. Alpha (L2 正規化參數)
st.sidebar.subheader('8. 正規化參數')
alpha = st.sidebar.number_input('Alpha (L2 正則化強度)', min_value=0.0001, max_value=1.0, value=0.0001, step=0.0001, format="%.4f")


# --- 模型訓練與評估 ---
st.header('模型訓練與結果')

# 顯示當前模型參數設定
st.write("當前模型參數設定：")
current_params = {
    "選擇的特徵": selected_features,
    "hidden_layer_sizes": hidden_layer_sizes,
    "activation": activation_function,
    "solver": solver,
    "alpha": alpha,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "learning_rate_init": learning_rate_init,
    "max_iter": max_iter,
    "early_stopping": early_stopping,
    "validation_fraction": validation_fraction if early_stopping else 'N/A (Early Stopping Disabled)',
    "n_iter_no_change": n_iter_no_change,
    "tol": tol
}
st.json(current_params)

# 執行訓練按鈕
if st.button('🚀 訓練模型'):
    if not selected_features:
        st.error("請至少選擇一個特徵才能訓練模型！")
    else:
        try:
            # 建立 MLP 模型實例
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation_function,
                solver=solver,
                alpha=alpha,
                batch_size=batch_size,
                learning_rate=learning_rate,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                early_stopping=early_stopping,
                validation_fraction=validation_fraction if early_stopping else 0.1, # 這裡設為 0.1 以符合預設值
                n_iter_no_change=n_iter_no_change,
                tol=tol,
                random_state=42
            )
            # 訓練模型，顯示進度
            with st.spinner('模型訓練中，請稍候...'):
                mlp.fit(X_train, y_train)
            st.success('✅ 模型訓練完成！')

            # 預測
            y_pred_train = mlp.predict(X_train)
            y_pred_test = mlp.predict(X_test)

            # --- 結果展示 ---
            st.subheader('模型評估結果')

            col1, col2 = st.columns(2)
            with col1:
                st.metric("訓練集準確率", f"{accuracy_score(y_train, y_pred_train):.4f}")
            with col2:
                st.metric("測試集準確率", f"{accuracy_score(y_test, y_pred_test):.4f}")

            st.subheader('分類報告 (測試集)')
            report = classification_report(y_test, y_pred_test, target_names=target_names, output_dict=True)
            st.json(report)

            st.subheader('混淆矩陣 (測試集)')
            cm = confusion_matrix(y_test, y_pred_test)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
            ax_cm.set_xlabel('預測標籤')
            ax_cm.set_ylabel('真實標籤')
            ax_cm.set_title('混淆矩陣')
            st.pyplot(fig_cm)
            plt.close(fig_cm) # 關閉圖形以釋放內存
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=target_names, yticklabels=target_names, ax=ax)
            ax.set_xlabel('預測標籤', fontproperties='SimHei')  # 設定軸標籤字型
            ax.set_ylabel('真實標籤', fontproperties='SimHei')
            ax.set_title('混淆矩陣', fontproperties='SimHei')  # 設定標題字型
            plt.setp(ax.get_xticklabels(), fontproperties='SimHei') # 設定刻度標籤字型
            plt.setp(ax.get_yticklabels(), fontproperties='SimHei')
            st.pyplot(fig)
            st.subheader('損失曲線')
            if hasattr(mlp, 'loss_curve_') and mlp.loss_curve_ is not None and len(mlp.loss_curve_) > 0:
                fig_loss, ax_loss = plt.subplots()
                ax_loss.plot(mlp.loss_curve_)
                ax_loss.set_xlabel('迭代次數')
                ax_loss.set_ylabel('損失')
                ax_loss.set_title('訓練損失曲線')
                st.pyplot(fig_loss)
                plt.close(fig_loss) # 關閉圖形以釋放內存
            else:
                st.info('目前選擇的優化器 (L-BFGS) 不會產生損失曲線。')

            # --- 互動式預測 ---
            st.subheader('💡 互動式預測')
            st.write(f'輸入您選擇的 {len(selected_features)} 個特徵值來預測 Iris 花的種類：')

            input_data = {}
            for feature in selected_features: # 根據選擇的特徵顯示輸入框
                # 計算原始數據的參考範圍
                feature_idx = all_feature_names.index(feature)
                min_val = X_train_full[feature].min() * loaded_scaler.scale_[feature_idx] + loaded_scaler.mean_[feature_idx]
                max_val = X_train_full[feature].max() * loaded_scaler.scale_[feature_idx] + loaded_scaler.mean_[feature_idx]
                avg_val = X_train_full[feature].mean() * loaded_scaler.scale_[feature_idx] + loaded_scaler.mean_[feature_idx]

                input_data[feature] = st.number_input(
                    f'輸入 {feature} (參考原始範圍: {min_val:.2f} ~ {max_val:.2f})',
                    value=float(avg_val),
                    step=0.1
                )

            if st.button('預測新資料'):
                # 將用戶輸入轉換為 DataFrame，只包含選擇的特徵
                input_df = pd.DataFrame([input_data], columns=selected_features)
                # 確保用戶輸入的特徵順序與訓練資料一致，並進行標準化
                # 注意：這裡的 StandardScaler 必須是訓練時用的那個，並且要處理未選擇特徵的情況
                # 為了簡化，這裡假設 loaded_scaler 是在所有原始特徵上訓練的
                # 我們需要創建一個與原始 X 結構相同的 DataFrame，未選擇的特徵設為平均值
                full_input_df = pd.DataFrame(columns=all_feature_names)
                for f in all_feature_names:
                    if f in input_df.columns:
                        full_input_df[f] = input_df[f]
                    else:
                        # 對於未選擇的特徵，填入其原始數據的平均值 (或 0，取決於 StandardScaler 的行為)
                        # 更穩健的做法是保存訓練時的特徵平均值
                        # 這裡為簡化，直接從原始數據中獲取平均值
                        original_mean = loaded_scaler.mean_[all_feature_names.index(f)]
                        full_input_df[f] = original_mean

                input_scaled = loaded_scaler.transform(full_input_df)

                prediction_proba = mlp.predict_proba(input_scaled)
                prediction_class = np.argmax(prediction_proba)

                st.write(f"預測的花的種類是: **{target_names[prediction_class]}**")
                st.write("各類別機率：")
                proba_df = pd.DataFrame({
                    '種類': target_names,
                    '機率': prediction_proba[0]
                })
                st.dataframe(proba_df.set_index('種類'))

        except Exception as e:
            st.error(f"模型訓練或預測過程中發生錯誤：\n\n`{e}`\n\n請檢查參數設定，或嘗試調整輸入值。")

st.markdown("---")
st.markdown("**資料來源**：[Scikit-learn Iris Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)")
st.markdown("**應用程式部署**：[Streamlit Cloud](https://streamlit.io/cloud)")
st.markdown("本應用由 Streamlit 構建，用於演示 MLP 模型參數調整與結果視覺化。")