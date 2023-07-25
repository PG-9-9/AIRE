// Your web app's Firebase configuration
var firebaseConfig = {
  apiKey: "AIzaSyA7C4qwXDACuVNNJjbyHpGi_mDzuvfYZQQ",
  authDomain: "aire-ed2c0.firebaseapp.com",
  databaseURL: "https://aire-ed2c0-default-rtdb.firebaseio.com",
  projectId: "aire-ed2c0",
  storageBucket: "aire-ed2c0.appspot.com",
  messagingSenderId: "885283015139",
  appId: "1:885283015139:web:9b0adc7e08b87443ddb8d6",
  measurementId: "G-65KNQLBSBX"
};
// Initialize Firebase
firebase.initializeApp(firebaseConfig);

    document.getElementById('register-form').addEventListener('submit', function(e){
      e.preventDefault();

      $("#myDiv").css("display", "block");

      //get user info
      var email = document.getElementById('email_up');
      var password = document.getElementById('pass_up');
      var fname = document.getElementById('fname_up');
      var lname = document.getElementById('lname_up');
      firebase.auth().createUserWithEmailAndPassword(email.value, password.value)
      .then(function(response) {
              firebase.database().ref('User_info/'+firebase.auth().currentUser.uid).set({
                  first_name: fname.value,
                  last_name: lname.value,
                  userid : firebase.auth().currentUser.uid
                }).then(() => {
                  firebase.auth().onAuthStateChanged(user => {
                    if(user) {
                      window.location = 'userInfoForm.html';
                    }
                  });
                });
      })
      .catch(function(error) {
        var errorCode = error.code;
        var errorMessage = error.message;
        console.log(errorCode);
        console.log(errorMessage);
      });
    });
