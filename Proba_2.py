#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, TimeDistributed, Conv1D, MaxPooling1D, Flatten, LayerNormalization, MultiHeadAttention, Attention, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import t

class DataPreprocessor:
    """
    Класс для предварительной обработки данных. Отфильтрует SKU, которые не продавались в течение последних пяти недель
    и те, которые появились менее четырех месяцев назад.
    """
    def __init__(self, data_path):
        """
        Конструктор класса, который загружает исходные данные из файла.
        :param data_path: путь к файлу с данными о продажах.
        """
        self.data = pd.read_csv(data_path)
        self.current_week = self.data['Week'].max()
        self.five_weeks_ago = self.current_week - pd.Timedelta(weeks=5)
        self.four_months_ago = self.current_week - pd.Timedelta(months=4)

    def filter_sku(self):
        """
        Метод для фильтрации SKU, которые не продавались в течение последних пяти недель или тех, которые появились менее чем четыре месяца назад.
        """
        index_to_drop = []
        for sku in self.data['SKU'].unique():
            rows = self.data.query("SKU == @sku & Week >= @self.five_weeks_ago")
            if len(rows) == 0 or rows['Продажи'].sum() == 0:
                index_to_drop.extend(self.data.query("SKU == @sku").index)

        self.data.drop(index_to_drop, inplace=True)

        index_to_drop = []
        for sku in self.data['SKU'].unique():
            rows = self.data.query("SKU == @sku & Week <= @self.four_months_ago")
            if len(rows) == 0:
                index_to_drop.extend(self.data.query("SKU == @sku").index)

        self.data.drop(index_to_drop, inplace=True)

    def create_time_series(self):
        """
        Метод для преобразования данных в временные ряды для каждого SKU.
        """
        time_series = {}
        for sku in self.data['SKU'].unique():
            rows = self.data.query("SKU == @sku")
            series = rows.set_index('Week')[['Продажи']]
            time_series[sku] = series

        return time_series

class ModelTrainer:
    """
    Класс для тренировки модели. Создает временные ряды, строит и тренирует модель.
    """
    def __init__(self, time_series, n_steps_in=96, n_steps_out=4):
        """
        Конструктор класса, который инициализирует временные ряды и параметры для обучения.
        :param time_series: словарь временных рядов, созданных в классе DataPreprocessor.
        :param n_steps_in: длина временного окна для обучения.
        :param n_steps_out: количество шагов вперед, которое нужно спрогнозировать.
        """
        self.time_series = time_series
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train, self.y_train, self.X_val, self.y_val = self.prepare_data()

    def prepare_data(self):
        """
        Метод для подготовки данных к тренировке. Формирует временные ряды для обучения и валидации.
        """
        X_train, y_train, X_val, y_val = [], [], [], []

        for sku, series in self.time_series.items():
            values = self.scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            n_features = 1
            X, y = [], []

            for i in range(len(values) - self.n_steps_in - self.n_steps_out + 1):
                X.append(values[i:i+self.n_steps_in])
                y.append(values[i+self.n_steps_in:i+self.n_steps_in+self.n_steps_out])

            X = np.array(X)
            y = np.array(y)

            split_idx = int(0.8 * len(X))
            X_train.append(X[:split_idx])
            y_train.append(y[:split_idx])
            X_val.append(X[split_idx:])
            y_val.append(y[split_idx:])

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        X_val = np.concatenate(X_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features)
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features)

        return X_train, y_train, X_val, y_val

    def build_model(self):
        """
        Метод для создания и настройки модели.
        """
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu',
                        input_shape=(self.n_steps_in, 1))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(MultiHeadAttention(num_heads=16, key_dim=64))
        model.add(LayerNormalization())
        model.add(Bidirectional(GRU(units=512, return_sequences=True)))
        model.add(TimeDistributed(Dense(256, activation='relu')))
        model.add(Dropout(0.25))
        model.add(Bidirectional(LSTM(units=256, return_sequences=True)))
        model.add(Dropout(0.15))
        model.add(Bidirectional(LSTM(units=128, return_sequences=False)))
        model.add(Dropout(0.15))
        model.add(Dense(units=self.n_steps_out))

        model.compile(optimizer='adam', loss='mse')

        return model

    def fit_model(self, model):
        """
        Метод для тренировки модели с использованием ранних остановок и снижения скорости обучения.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=60, min_lr=0.0000001)

        history = model.fit(self.X_train, self.y_train, epochs=1200, validation_data=(self.X_val, self.y_val),
                           callbacks=[early_stopping, reduce_lr], batch_size=256)

        return history

    def plot_history(self, history):
        """
        Метод для визуализации графика потерь во время тренировки.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict_future_sales(self, model, future_X):
        """
        Метод для генерации прогнозов на основании данных.
        """
        future_sales = model.predict(future_X)
        future_sales = self.scaler.inverse_transform(future_sales)
        return future_sales

class SeasonalityAnalyzer:
    """
    Класс для анализа сезонности в данных.
    """
    def __init__(self, time_series):
        self.time_series = time_series

    def decompose(self):
        """
        Метод для декомпозиции временных рядов на компоненты: сезонность, тренд и остаточные значения.
        """
        decomposition_results = {}
        for sku, series in self.time_series.items():
            result = seasonal_decompose(series, model='additive', period=52)
            decomposition_results[sku] = {
                'observed': result.observed,
                'trend': result.trend,
                self.time_series[sku]['seasonal']: result.seasonal,
                'residual': result.resid
            }
        return decomposition_results

class ConfidenceIntervalCalculator:
    """
    Класс для расчета доверительных интервалов.
    """
    def __init__(self, predictions, actual_values, alpha=0.05):
        self.predictions = predictions
        self.actual_values = actual_values
        self.alpha = alpha

    def calculate_confidence_intervals(self):
        """
        Метод для вычисления нижних и верхних границ доверительных интервалов.
        """
        errors = self.actual_values - self.predictions
        mean_error = np.mean(errors)
        std_error = np.std(errors, ddof=1)
        n = len(errors)
        margin_of_error = t.ppf(1 - self.alpha/2, n-1) * std_error / np.sqrt(n)
        lower_bound = mean_error - margin_of_error
        upper_bound = mean_error + margin_of_error
        return lower_bound, upper_bound

# Загрузка и предобработка данных
data_preprocessor = DataPreprocessor('sales_data.csv')
data_preprocessor.filter_sku()
time_series = data_preprocessor.create_time_series()

# Обучение модели
model_trainer = ModelTrainer(time_series)
model = model_trainer.build_model()
history = model_trainer.fit_model(model)
model_trainer.plot_history(history)

# Анализ сезонности
seasonality_analyzer = SeasonalityAnalyzer(time_series)
decomposition_results = seasonality_analyzer.decompose()

# Прогнозирование будущих продаж
future_X = None  #данные для будущих 4 недель
future_sales = model_trainer.predict_future_sales(model, future_X)

# Расчет доверительного интервала
confidence_interval_calculator = ConfidenceIntervalCalculator(future_sales, actual_values=None, alpha=0.05)
lower_bound, upper_bound = confidence_interval_calculator.calculate_confidence_intervals()

# Сохранение результатов
np.savetxt('predicted_sales.csv', future_sales, delimiter=',', fmt='%f')

