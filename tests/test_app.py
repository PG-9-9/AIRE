import pyrebase
import json

def test_index(app,client):
      
        firebaseConfig = {
    "apiKey": "",
    "authDomain": "",
    "databaseURL": "",
    "projectId": "",
    "storageBucket": "",
    "messagingSenderId": "",
    "appId": "",
    "measurementId": ""
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
