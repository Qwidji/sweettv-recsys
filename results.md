# Results: offline validation

Протокол: train = ISO-недели 11–20, valid = 21–30. Пул кандидатов: tv_show_id != 0 из `export_arh_21-30-final.csv`. MAP@5 со знаменателем min(5, |truth|). Повторы разрешены; холодные пользователи и добивка до 5 берутся из popularity (лучший вариант по valid). Seed = 42.

Пользователей в оценке: 4799 (все с непустым truth), 1246 из них тестовые.

| Метод | Параметры | MAP@5 (все) | MAP@5 (тестовые) | Время, с |
|---|---|---|---|---|
| Popularity | unique viewers, weeks 11-20 | 0.61945 | 0.73167 | 0.1 |
| Popularity | unique viewers, weeks 19-20 | 0.62279 | 0.73417 | 0.0 |
| History | sum overlap desc, pad popularity | 0.56280 | 0.67345 | 0.2 |
| ALS | factors=32, reg=0.01, iter=20 | 0.48842 | 0.54516 | 1.6 |
| ALS | factors=32, reg=0.1, iter=20 | 0.48941 | 0.54618 | 1.5 |
| ALS | factors=64, reg=0.01, iter=20 | 0.41814 | 0.45097 | 2.6 |
| ALS | factors=64, reg=0.1, iter=20 | 0.41907 | 0.45197 | 2.6 |
| Content | TF-IDF cat+genres+director+actors | 0.17469 | 0.19350 | 4.6 |
| Hybrid | RRF k0=10, w_hist=1, w_als=0.5, w_pop=0.5 (ALS factors=32, reg=0.1) | 0.65976 | 0.77418 | 4.3 |

Выбор лучшего ALS и весов Hybrid сделан по тому же valid, поэтому их оценки оптимистичны.

## Hybrid: топ-5 конфигураций

| MAP@5 (все) | MAP@5 (тестовые) | k0 | w_als | w_pop |
|---|---|---|---|---|
| 0.65976 | 0.77418 | 10 | 0.5 | 0.5 |
| 0.65482 | 0.76629 | 60 | 0.5 | 0.5 |
| 0.65409 | 0.75898 | 10 | 1 | 0.5 |
| 0.65234 | 0.76147 | 60 | 1 | 0.5 |
| 0.65213 | 0.76890 | 10 | 0.25 | 0.5 |

## Данные

- train interactions: 1427078, user-show pairs: 562815, users: 4833
- pool (tv_show_id != 0 in export_arh_21-30-final.csv): 6363 shows; in train items: 2154
- eval users (non-empty valid truth): 4799; cold (no train): 57
- test users with non-empty valid truth: 1246 of 1260
- truth user-show pairs: 466900; in pool: 0.9998

## Unit-тест MAP@5

```
truth=[1, 2, 3] recs=[1, 4, 2, 5, 6] AP@5=0.5556 expected=0.5556 OK
truth=[7] recs=[3, 7, 8, 9, 10] AP@5=0.5000 expected=0.5000 OK
truth=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] recs=[0, 1, 2, 3, 4] AP@5=1.0000 expected=1.0000 OK
truth=[1, 2] recs=[3, 4, 5, 6, 1] AP@5=0.1000 expected=0.1000 OK
truth=[1, 2] recs=[9, 8, 7, 6, 5] AP@5=0.0000 expected=0.0000 OK
MAP@5 over the 5 cases = 0.4311
```

## Окружение

- Python 3.13.5, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, implicit 0.7.3
- Apple M1, 8 CPU, 8 GB RAM, macOS-26.1-arm64-arm-64bit-Mach-O
