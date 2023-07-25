// Your web app's Firebase configuration
  // For Firebase JS SDK v7.20.0 and later, measurementId is optional
  var firebaseConfig = {
    apiKey: "AIzaSyA7C4qwXDACuVNNJjbyHpGi_mDzuvfYZQQ",
    authDomain: "aire-ed2c0.firebaseapp.com",
    projectId: "aire-ed2c0",
    storageBucket: "aire-ed2c0.appspot.com",
    messagingSenderId: "885283015139",
    appId: "1:885283015139:web:9b0adc7e08b87443ddb8d6",
    measurementId: "G-65KNQLBSBX"
  };
  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);
  firebase.analytics();

  firebase.auth().onAuthStateChanged(function(user){
      if(user){
          var current_user_id = firebase.auth().currentUser.uid;
          firebase.database().ref('User_info/'+current_user_id).on('value', function(snapshot){
            var fname_to_be_displayed = snapshot.val().first_name;
            var lname_to_be_displayed = snapshot.val().last_name;
            document.getElementById("name_of_user").innerHTML = fname_to_be_displayed + " " + lname_to_be_displayed;
            document.getElementById("name_of_user_2").innerHTML = fname_to_be_displayed + " " + lname_to_be_displayed;
            const container = document.getElementById('usercardlayout');
            //create card element
            const card = document.createElement('div');
            card.classList = 'card-body';

            //constructing card
            const content = `
            <div class="row">
              <div class="card-container>
                  <section class="page-contain">
                      <a onclick="togglePopupone()" class="data-card">
                      <h3 class="hthree">Tier ${snapshot.val().tier}</h3>
                      <h3 class="hfour">INTERESTED SECTOR:</h3>
                      <h5 class="hfive">${snapshot.val().interested_sector}</h5>
                      </a>
                  </section>
              </div>
              <div class="card-container>
                <section class="page-contain">
                  <a onclick="togglePopuptwo()" class="data-card-two">
                  <h3 class="hthree">RISK SCORE: ${snapshot.val().risk_score}<h3>
                  <h3 class="hfour">INVESTMENTS MADE:</h3>
                  <h5 class="hfive">${snapshot.val().stocks_already}</h5>
                  </a>
                </section>
              </div>
            </div>
                `;

                //append newly created card element to the container
                container.innerHTML += content;

          })
      }

  });
  function signOut(){
    firebase.auth().signOut();
    firebase.auth().onAuthStateChanged(user => {
        if(user) {
          window.location = 'userInfoForm.html';
        }
        else{
            window.location = '/';
        }
      });
  }

  function togglePopupone(){
    document.getElementById("popup-1").classList.toggle("active");

  }

  function togglePopuptwo(){
    document.getElementById("popup-2").classList.toggle("active");

  }

  