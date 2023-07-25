import pyrebase
import json

def test_index(app,client):
      
        firebaseConfig = {
    "apiKey": "AIzaSyA7C4qwXDACuVNNJjbyHpGi_mDzuvfYZQQ",
    "authDomain": "aire-ed2c0.firebaseapp.com",
    "databaseURL": "https://aire-ed2c0-default-rtdb.firebaseio.com/",
    "projectId": "aire-ed2c0",
    "storageBucket": "aire-ed2c0.appspot.com",
    "messagingSenderId": "885283015139",
    "appId": "1:885283015139:web:9b0adc7e08b87443ddb8d6",
    "measurementId": "G-65KNQLBSBX"
    };
        firebase = pyrebase.initialize_app(firebaseConfig)

        db = firebase.database()
        users = db.child("User_info").get()
        if users is not None:
                del app
                res = client.get('/signin.html')
                assert res.status_code == 200
                expected = {'sucess': '1'}
                assert expected == json.loads(res.get_data(as_text=True))
