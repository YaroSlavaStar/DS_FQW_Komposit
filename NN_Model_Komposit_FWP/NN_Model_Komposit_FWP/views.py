
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from flask import render_template, request
from NN_Model_Komposit_FWP import app

@app.route('/', methods=['POST', 'GET'])

@app.route('/home', methods=['POST', 'GET'])
def home():
    if request.method == 'GET':
        return render_template(
            'index.html',
            title='Главная',
            year=datetime.now().year,
        )
    
    if request.method == 'POST': 
        with open('nn_model_komposit') as nn_model_komposit:
            nn_model = tf.keras.models.load_model(nn_model_komposit)
        cols = ['Плотность, кг/м3', 
                'модуль упругости, ГПа',
                'Количество отвердителя, м.%', 
                'Содержание эпоксидных групп,%_2',
                'Температура вспышки, С_2', 
                'Поверхностная плотность, г/м2',
                'Модуль упругости при растяжении, ГПа', 
                'Прочность при растяжении, МПа',
                'Потребление смолы, г/м2', 
                'Угол нашивки, град', 
                'Шаг нашивки',
                'Плотность нашивки']
        param_lst = [(request.form.get('param1')),
                     (request.form.get('param2')),
                     (request.form.get('param3')),
                     (request.form.get('param4')),
                     (request.form.get('param5')),
                     (request.form.get('param6')),
                     (request.form.get('param7')),
                     (request.form.get('param8')),
                     (request.form.get('param9')),
                     (request.form.get('param10')),
                     (request.form.get('param11')),
                     (request.form.get('param12'))]
        final = np.array(param_lst)
        data_unseen = pd.DataFrame([final], columns=cols)
        prediction = nn_model.predict([[data_unseen]])
        return render_template('index.html', prediction)

@app.route('/contact')
def contact():
    return render_template(
        'contact.html',
        title='Контакты',
        year=datetime.now().year,
        message='Мои контактные данные.'
    )

@app.route('/about')
def about():
    return render_template(
        'about.html',
        title='Описание',
        year=datetime.now().year,
        message='Описание приложения.'
    )
