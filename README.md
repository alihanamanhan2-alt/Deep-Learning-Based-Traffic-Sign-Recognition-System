# Deep-Learning-Based-Traffic-Sign-Recognition-System

GTSRB жол белгілерін тану жүйесі. Жоба PyTorch негізінде custom CNN және ResNet18 архитектураларын біріктіреді, 43 сыныпты таниды және Streamlit интерфейсін ұсынады.

## Құрылым

```text
app.py                 Streamlit интерфейсі
train.py               CNN/ResNet18 оқыту
 evaluate.py           тест деректерімен бағалау
predict.py             жекелей және пакеттік болжау
compare_models.py      модельдерді салыстыру
src/dataset.py         GTSRB деректері мен DataLoader
src/model.py           CNN және ResNet18
src/preprocessing.py   resize және augmentation
src/metrics.py         report, confusion matrix, графиктер
COURSEWORK.md          қазақша курстық жұмыс мәтіні
```

## Орнату

```bash
python -m venv venv
# Windows
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GTSRB деректерін `data/raw/GTSRB` каталогына placement нұсқауына сай орналастырыңыз.

## Оқыту

```bash
python train.py --model cnn --epochs 20
python train.py --model resnet18 --epochs 20 --pretrained
```

## Бағалау

```bash
python evaluate.py --model cnn --data-dir data/raw/GTSRB
python evaluate.py --model resnet18 --data-dir data/raw/GTSRB
python compare_models.py
```

## Қосымшаны іске қосу

```bash
streamlit run app.py
```

## Тест

```bash
pytest -q
```

Нақты метрикалар GTSRB деректері және оқытылған `.pth` файлдары болған жағдайда ғана шығарылады. Жоба демонстрациялық немесе ойдан жасалған accuracy нәтижелерін жасамайды.
