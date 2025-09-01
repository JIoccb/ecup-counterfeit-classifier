# -*- coding: utf-8 -*-
"""
tabular_analyze.py
==================

Функции для инференса по табличным признакам (одна строка из исходного DataFrame).
Поддерживает различные варианты sklearn-пайплайнов:
- с `predict_proba` (предпочтительно),
- с `decision_function` (преобразуем score через сигмоиду),
- только с `predict` (фоллбек к 0/1).

Особенности
-----------
- Сохраняется порядок колонок (если у пайплайна есть `feature_names_in_`, используем его).
- Если часть ожидаемых колонок отсутствует в строке — они пропускаются (это лучше,
  чем падение, но может ухудшить качество; следите за согласованностью признаков).
"""

import math  # оставлен для совместимости, может не использоваться напрямую
import numpy as np


def _sigmoid(x):
    """
    Скалярная сигмоида для приведения score -> [0, 1].
    """
    return 1.0 / (1.0 + np.exp(-x))


def _as_2d(df_row, cols):
    """
    Привести серию к DataFrame формата (1, len(cols)) в указанном порядке колонок.

    Параметры
    ---------
    df_row : pandas.Series
        Одна строка исходного DataFrame.
    cols : list[str]
        Список имён признаков в нужном порядке.

    Возвращает
    ----------
    pandas.DataFrame
        Однострочный DataFrame в порядке колонок `cols` (исключая отсутствующие).
    """
    # Важно: используем .to_frame().T, чтобы сохранить 2D-формат
    return df_row[cols].to_frame().T


def predict_tabular_prob(row_df, pipeline, expected_cols=None):
    """
    Получить вероятность положительного класса из sklearn-пайплайна/модели.

    Параметры
    ---------
    row_df : pandas.Series
        Одна строка из исходного DataFrame (все признаки доступны по индексам).
    pipeline : sklearn-like object
        Обученный пайплайн/модель (загруженный из pickle).
    expected_cols : Iterable[str] | None, optional
        Явный список колонок в ожидаемом порядке. Если None:
        - используем pipeline.feature_names_in_, если он есть,
        - иначе берём колонки прямо из row_df в имеющемся порядке.

    Возвращает
    ----------
    float
        Вероятность положительного класса (в интервале [0, 1]).
    """
    import pandas as pd

    # Определяем ожидаемые колонки и сохраняем порядок
    if expected_cols is None and hasattr(pipeline, "feature_names_in_"):
        expected_cols = list(pipeline.feature_names_in_)
    elif expected_cols is None:
        expected_cols = list(row_df.index)

    # Отфильтровываем только те колонки, которые реально есть в строке
    cols_present = [c for c in expected_cols if c in row_df.index]
    X = _as_2d(row_df, cols_present)

    # 1) Предпочтительный путь: predict_proba
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)

        # Бинарный случай: столбец с индексом 1 — положительный класс
        if proba.shape[1] == 2:
            return float(proba[0, 1])

        # Мультикласс (редко в этой задаче): берём максимум как приблизительную оценку «позитива»
        return float(proba[0].max())

    # 2) decision_function -> сигмоида
    if hasattr(pipeline, "decision_function"):
        score = pipeline.decision_function(X)

        # Приводим к (N,) если получился (N, 1)
        if np.ndim(score) == 2 and score.shape[1] == 1:
            score = score.ravel()

        return float(_sigmoid(score[0]))

    # 3) Фоллбек: predict -> {0,1}
    pred = pipeline.predict(X)
    return float(pred[0])
