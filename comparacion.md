| Modelo            | Parámetros principales                                                    | Mejor F1 (CV) | Accuracy (test) | Clase        | Precision | Recall | F1-score | Support |
|-------------------|--------------------------------------------------------------------------|---------------|-----------------|--------------|-----------|--------|----------|---------|
| Random Forest     | n_estimators=200, min_samples_split=2, min_samples_leaf=1, max_depth=30, bootstrap=False | 0.8770        | 0.78            | 0            | 0.84      | 0.87   | 0.86     | 1035    |
|                   |                                                                          |               |                 | 1            | 0.60      | 0.54   | 0.57     | 374     |
|                   |                                                                          |               |                 | Macro Avg    | 0.72      | 0.71   | 0.71     | 1409    |
|                   |                                                                          |               |                 | Weighted Avg | 0.78      | 0.78   | 0.78     | 1409    |
| Gradient Boosting | subsample=0.8, n_estimators=100, min_samples_split=10, min_samples_leaf=4, max_depth=7, learning_rate=0.05 | 0.8513        | 0.78            | 0            | 0.86      | 0.83   | 0.85     | 1035    |
|                   |                                                                          |               |                 | 1            | 0.57      | 0.62   | 0.59     | 374     |
|                   |                                                                          |               |                 | Macro Avg    | 0.71      | 0.73   | 0.72     | 1409    |
|                   |                                                                          |               |                 | Weighted Avg | 0.78      | 0.78   | 0.78     | 1409    |
