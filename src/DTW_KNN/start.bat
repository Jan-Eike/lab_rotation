set root=C:\ProgramData\Miniconda3

call %root%\Scripts\activate.bat

python save_data.py
python main.py --use_saved_k=True --save_classification=True --test_length_start=0 --test_length_end=100 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=100 --test_length_end=200 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=200 --test_length_end=300 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=300 --test_length_end=400 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=400 --test_length_end=500 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=500 --test_length_end=600 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=600 --test_length_end=700 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=700 --test_length_end=800 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=800 --test_length_end=900 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=900 --test_length_end=1000 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=1000 --test_length_end=1100 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=1100 --test_length_end=1200 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=1200 --test_length_end=1300 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=1300 --test_length_end=1400 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=1400 --test_length_end=1500 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=1500 --test_length_end=1600 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=1600 --test_length_end=1700 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=1700 --test_length_end=1800 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=1800 --test_length_end=1900 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=1900 --test_length_end=2000 --result=False
timeout /t 1
python main.py --use_saved_k=True --save_classification=True --test_length_start=2000 --test_length_end=-1 --result=True
pause