import torch
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from model_tester import ModelTester
from model_evaluator import ModelEvaluator
from models import SimpleModel


def main():
    # Загрузка данных
    print("Загрузка данных...")
    data_loader = DataLoader()
    data, target = data_loader.load_data()
    print("Данные успешно загружены.")

    # Предварительная обработка данных
    print("Предварительная обработка данных...")
    data_preprocessor = DataPreprocessor()
    train_data, test_data, train_target, test_target = data_preprocessor.preprocess_data(data, target)
    print("Данные успешно обработаны.")

    # Обучение моделей
    print("Обучение моделей...")
    # Создание экземпляра класса ModelTrainer
    model_trainer = ModelTrainer(train_data, train_target, X_val=None, y_val=None)


    print("Создание моделей...")
    # Создание модели
    model = SimpleModel()

    print("Обучение модели...")
    # Обучение модели
    model_trainer.train_model(model, learning_rate=0.01, num_epochs=100)

    print("Получение обученной моделм...")
    # Получение обученной модели
    trained_model = model_trainer.get_model()
    print("Модели успешно обучены.")

    # Тестирование моделей
    print("Тестирование моделей...")
    model_tester = ModelTester(test_data, test_target)
    test_data_tensor = torch.tensor(test_data.values, dtype=torch.float32)
    test_target_tensor = torch.tensor(test_target.values, dtype=torch.float32)
    test_results = model_tester.test_models({'trained_model': trained_model}, test_data_tensor, test_target_tensor)

    print("Модели успешно протестированы.")

    # Оценка моделей
    print("Оценка моделей...")
    model_evaluator = ModelEvaluator(test_data_tensor, test_target_tensor)
    evaluation_results = model_evaluator.evaluate_model(trained_model)
    print("Модели успешно оценены.")

    # Вывод результатов
    print("Результаты оценки моделей:")
    print(f"Среднеквадратичное отклонение: {evaluation_results['mse']}")
    print(f"Коэффициент детерминации: {evaluation_results['r2']}")
    print(f"Среднее абсолютное отклонение: {evaluation_results['mae']}")
    print()


if __name__ == "__main__":
    main()
