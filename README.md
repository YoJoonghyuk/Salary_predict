# Salary_predict

Проект по предсказанию зарплат на основе данных резюме с использованием машинного обучения.

Проект состоит из трех основных этапов: **парсинг** исходного CSV в числовые признаки, **обучение** модели линейной регрессии и **предсказание** зарплат на основе подготовленных признаков.
##  Структура Репозитория

* salary_predictor/
* ├── app.py
* ├── parse_data.py
* ├── train_model.py
* ├── .gitignore
* ├── README.md
* ├── docs/ 
* ├── src/
* │   ├── base.py
* │   ├── loaders.py
* │   ├── npy_loader.py
* │   ├── predictor_handler.py
* │   ├── output.py
* │   └── transformation.py
* └── utils/
*  ├── age_parser.py
*  ├── experience_parser.py
*  ├── salary_parser.py
*  ├── city_parser.py
*  ├── helpers.py
*  ├── transformer_utils.py
*  └── outlier_remover.py

##  Установка Зависимостей
Убедитесь, что у вас установлен Python 3.8+.
Установите необходимые библиотеки, используя pip:
                   ` pip install pandas numpy scikit-learn joblib`
  ##  Запуск Проекта
* Шаг 1: **Парсинг данных**

Этот скрипт читает исходный hh.csv, извлекает необходимые признаки (пол, возраст, опыт, город, должность, зарплата), масштабирует числовые признаки, векторизует текстовые и сохраняет результат в data/x_data.npy (признаки) и data/y_data.npy (целевые значения). Также будут сохранены обученные scaler.pkl и vectorizer.pkl в папку resources/.
  
 `python parse_data.py data/hh.csv`

* Шаг 2: **Обучение модели**
                            
Этот скрипт загружает x_data.npy и y_data.npy, обучает модель LinearRegression, оценивает её с помощью метрик $R^2$ и MAE, а затем сохраняет обученную модель в resources/model.pkl.

   `python train_model.py`

* Шаг 3: **Предсказание зарплат** 
                           
Этот скрипт загружает x_data.npy (или любой другой .npy файл с аналогично подготовленными признаками), загружает обученную модель из resources/model.pkl и выдает список предсказанных зарплат (List[float]) в консоль.

   `python app.py data/x_data.npy`
