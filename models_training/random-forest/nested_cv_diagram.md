# Random Forest Nested CV Flow

```mermaid
flowchart TB
    A[Load config from JSON] --> B[Load and sort dataset by year / round]
    B --> C[Build features and encode target]
    C --> D[Create race-level TimeSeriesSplit folds]

    D --> E{Outer fold k}
    E --> F[Outer train races]
    E --> G[Outer test races]

    F --> H[Inner race-level TimeSeriesSplit]
    H --> I[RandomizedSearchCV over hyperparameters]
    I --> J[Pick best params on inner validation races]
    J --> K[Refit best model on all outer training races]
    K --> L[Evaluate on outer test races]

    L --> M[Store macro F1, balanced accuracy, best params, feature importances]
    M --> N[Repeat for all outer folds]

    N --> O[Aggregate outer-fold metrics]
    N --> P[Pool all outer test predictions]
    N --> Q[Fresh RandomizedSearchCV on full dataset]
    Q --> R[Retrain final model with best full-data params]
    R --> S[Save model and nested CV summary]

    style A fill:#f8f9fa,stroke:#333,stroke-width:1px
    style D fill:#e8f4ff,stroke:#2b6cb0,stroke-width:1px
    style I fill:#fff3cd,stroke:#b7791f,stroke-width:1px
    style L fill:#fde2e2,stroke:#c53030,stroke-width:1px
    style R fill:#e6ffed,stroke:#2f855a,stroke-width:1px
```

## Reading guide

- The **outer loop** estimates performance on future races.
- The **inner loop** chooses hyperparameters using only the outer training races.
- The final model is selected with a fresh search on all data after nested CV finishes.
