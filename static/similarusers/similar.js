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

  firebase.auth().onAuthStateChanged(function(user){
    if(user){
        var current_user_id = firebase.auth().currentUser.uid;
        firebase.database().ref('User_info/'+current_user_id).on('value', function(snapshot){
          var fname_to_be_displayed = snapshot.val().first_name;
          var lname_to_be_displayed = snapshot.val().last_name;
          document.getElementById("name_of_user").innerHTML = fname_to_be_displayed + " " + lname_to_be_displayed;
          document.getElementById("name_of_user_2").innerHTML = fname_to_be_displayed + " " + lname_to_be_displayed;
        })
    }

});
function signOut(){
  firebase.auth().signOut();
  alert("SignOut");
  firebase.auth().onAuthStateChanged(user => {
      if(user) {
        window.location = 'userInfoForm.html';
      }
      else{
          window.location = 'index.html'
      }
    });
}

//fucntion to calculate similar users
firebase.auth().onAuthStateChanged(function(user){
    if(user){
        var current_user_id = firebase.auth().currentUser.uid;
        firebase.database().ref('User_info/'+current_user_id).on('value', function(snapshot){
            var user_tier = snapshot.val().tier;
            var user_interest = snapshot.val().interested_sector;
            firebase.database().ref('User_info').orderByChild("tier").equalTo(user_tier).on("child_added", (snap) => {
                var simusers = snap.val();
                console.log(simusers);
                if (simusers.userid == current_user_id){

                }
                else{

                    if(simusers.interested_sector == user_interest){
                        const container = document.getElementById('cardlayout');
                            //create card element
                            const card = document.createElement('div');
                            card.classList = 'card-body';

                            //constructing card
                            const content = `
                            <div class="card-container>
                                <section class="page-contain">
                                    <a class="data-card-two">
                                    <h3 class="hthree">${simusers.first_name} ${simusers.last_name}</h3>
                                    <h4 class="hfour">Tier ${simusers.tier}</h4>
                                    <h4 class="hfour">INTERESTED SECTOR:</h4>
                                    <h5 class="hfive">${simusers.interested_sector}</h5>
                                    <h4 class="hfour">INVESTMENTS MADE:</h4>
                                    <h5 class="hfive">${simusers.stocks_already}</h5>
                                    <h4 class="hfour">INVESTMENTS MADE:</h4>
                                    <h5 class="hfive">${simusers.risk_score}</h5>
                                    </a>
                            </section>
                            </div>
                                `;

                                //append newly created card element to the container
                                container.innerHTML += content;
                        }
                        else{
                            const container_2 = document.getElementById('cardlayout_2');
                            //create card element
                            const card = document.createElement('div');
                            card.classList = 'card-body';

                            //constructing card
                            const content_2 = `
                                    
                            <div class="card-container>
                                <section class="page-contain">
                                    <a class="data-card">
                                    <h3 class="hthree">${simusers.first_name} ${simusers.last_name}</h3>
                                    <h4 class="hfour">Tier ${simusers.tier}</h4>
                                    <h4 class="hfour">INTERESTED SECTOR:</h4>
                                    <h5 class="hfive">${simusers.interested_sector}</h5>
                                    <h4 class="hfour">INVESTMENTS MADE:</h4>
                                    <h5 class="hfive">${simusers.stocks_already}</h5>
                                    <h4 class="hfour">INVESTMENTS MADE:</h4>
                                    <h5 class="hfive">${simusers.risk_score}</h5>
                                    </a>
                            </section>
                            </div> 
                                `;

                                //append newly created card element to the container
                                container_2.innerHTML += content_2;
                        }
                }
                
            });
        })
    }

});