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


var sente = "yomama";

//referencing user_info 
firebase.auth().onAuthStateChanged(function(user){
  if(user) {
    var current_user_id = firebase.auth().currentUser.uid;
    var user_infoRef = firebase.database().ref('User_info/'+current_user_id);
    document.getElementById('infoForm').addEventListener('submit', submitForm);

    //submit form
    function submitForm(e){
        e.preventDefault();

        $("#myDiv").css("display", "block");

        //get values
        var age = getInputVal('age')
        var country = getInputVal('country')
        var yearly_income= getInputVal('yearly_income')
        var yearly_investments = getInputVal('yearly_investments')
        var occupied_sector = getInputVal('occupied_sector')
        var pref_ror = getInputVal('pref_ror')
        var interested_sector = getInputVal('interested_sector')
        var ques_2 = getInputVal('ques_2')
        var ques_3 = getInputVal('ques_3')
        var ques_4 = getInputVal('ques_4')
        var ques_5 = getInputVal('ques_5')
        var ques_6 = getInputVal('ques_6')
        var ques_7 = getInputVal('ques_7')
        var ques_8 = getInputVal('ques_8')
        var ques_9 = getInputVal('ques_9')
        var ques_10 = getInputVal('ques_10')
        var ques_11 = getInputVal('ques_11')
        var ques_12 = getInputVal('ques_12')
        var stocks_already = getInputVal('stocks_already')
        var extra_message = getInputVal('extra_message')
        //calling send to firebase fucntion
        saveUserInfo(age, country, yearly_income, yearly_investments, occupied_sector, pref_ror, interested_sector, ques_2, ques_3, ques_4, ques_5, ques_6, ques_7, ques_8, ques_9, ques_10, ques_11, ques_12, stocks_already, extra_message);
    }

    //function to get form values
    function getInputVal(id){
        return document.getElementById(id).value;
    }

    //save to firebase
    function saveUserInfo(age, country, yearly_income, yearly_investments, occupied_sector, pref_ror, interested_sector, ques_2, ques_3, ques_4, ques_5, ques_6, ques_7, ques_8, ques_9, ques_10, ques_11, ques_12, stocks_already, extra_message){
        user_infoRef.update({
            age: age,
            country: country,
            yearly_income: yearly_income,
            yearly_investments: yearly_investments,
            occupied_sector: occupied_sector,
            pref_ror: pref_ror,
            interested_sector: interested_sector,
            ques_2: ques_2,
            ques_3: ques_3,
            ques_4: ques_4,
            ques_5: ques_5,
            ques_6: ques_6,
            ques_7: ques_7,
            ques_8: ques_8,
            ques_9: ques_9,
            ques_10: ques_10,
            ques_11: ques_11,
            ques_12: ques_12,
            stocks_already: stocks_already,
            extra_message: extra_message

        }).then(() => {
          window.location = 'dashboard.html';
        })
    }
    firebase.database().ref('User_info/'+current_user_id).on('value', function(snapshot){
      var fname_to_be_displayed = snapshot.val().first_name;
      var lname_to_be_displayed = snapshot.val().last_name;
      document.getElementById("sampletry").innerHTML = fname_to_be_displayed + " " + lname_to_be_displayed;
    })
    
  }
});

