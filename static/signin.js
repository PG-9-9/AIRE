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

  document.getElementById('sign_in_form').addEventListener('submit', function(e){
    e.preventDefault();

    $("#myDiv").css("display", "block");
    //get user info
    var email = document.getElementById('email_in');
    var password = document.getElementById('pass_in');
    firebase.auth().signInWithEmailAndPassword(email.value, password.value)
    .then(function(response) {
                firebase.auth().onAuthStateChanged(user => {
                  if(user) {
                    window.location = 'dashboard.html';
                    var current_user_id = firebase.auth().currentUser.uid;
                    firebase.database().ref('User_info/'+current_user_id).on('value', function(snapshot){
                      var risk_score = snapshot.val().risk_score;
                      $.ajax({
                        url: '/signin.html',
                        type: 'POST',
                        data: {
                          'risk_score': risk_score,
                        },
                        success: function(response){
                          console.log(response);
                        }
                      })
                    });

                  }
                  else {
                    window.location = 'index.html';
                  }
                });
    })
    .catch(function(error) {
      var errorCode = error.code;
      var errorMessage = error.message;
      console.log(errorCode);
      console.log(errorMessage);
    });
  });
