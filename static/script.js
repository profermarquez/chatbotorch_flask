

const data = new function(){
    //console.log('feching..');
    fetch("./intents.json")
    .then((res) => {
        return res.json();
      })
      .then((data) => 
        {
            //sucess
            var crudApp2 =new function(){
                this.arreglo=[];
                this.col = [];

                
            
                this.createTable = function () {
                    //console.log("e");
                    for (var i = 0; i < this.arreglo.length; i++) {
            
                        for (var key in this.arreglo[i]) 
                            {
                                if (this.col.indexOf(key) === -1) {
                                    this.col.push(key);
                                }
                            }
                    }//console.log(this.col);

                    var table = document.createElement('table');
                    console.log(table);
                    table.setAttribute('id', 'jsonTable');  
                    var tr = table.insertRow(-1);

                    for (var h = 0; h < this.col.length; h++) {
                        // Add table header.
                        var th = document.createElement('th');
                        th.innerHTML = this.col[h].replace('_', ' ');
                        tr.appendChild(th);
                    }

                    for (var i = 0; i < this.arreglo.length; i++) {

                        tr = table.insertRow(-1);           // Create a new row.
        
                        for (var j = 0; j < this.col.length; j++) {
                            var tabCell = tr.insertCell(-1);
                            tabCell.innerHTML = this.arreglo[i][this.col[j]];
                        }
        
                        // Dynamically create and add elements to table cells with events.
        
                        this.td = document.createElement('td');
        
                        // *** CANCEL OPTION.
                        tr.appendChild(this.td);
                        var lblCancel = document.createElement('label');
                        lblCancel.innerHTML = '✖';
                        lblCancel.setAttribute('onclick', 'crudApp2.Cancel(this)');
                        lblCancel.setAttribute('style', 'display:none;');
                        lblCancel.setAttribute('title', 'Cancel');
                        lblCancel.setAttribute('id', 'lbl' + i);
                        this.td.appendChild(lblCancel);
        
                        // *** SAVE.
                        tr.appendChild(this.td);
                        var btSave = document.createElement('input');
        
                        btSave.setAttribute('type', 'button');      // SET ATTRIBUTES.
                        btSave.setAttribute('value', 'Save');
                        btSave.setAttribute('id', 'Save' + i);
                        btSave.setAttribute('style', 'display:none;');
                        btSave.setAttribute('onclick', 'Save(this)');       // ADD THE BUTTON's 'onclick' EVENT.
                        this.td.appendChild(btSave);
        
                        // *** UPDATE.
                        tr.appendChild(this.td);
                        var btUpdate = document.createElement('input');
        
                        btUpdate.setAttribute('type', 'button');    // SET ATTRIBUTES.
                        btUpdate.setAttribute('value', 'Update');
                        btUpdate.setAttribute('id', 'Edit' + i);
                        btUpdate.setAttribute('disabled','true') 
                        btUpdate.setAttribute('style', 'background-color:#44CCEB;');
                        btUpdate.setAttribute('onclick', 'Update(this)');   // ADD THE BUTTON's 'onclick' EVENT.
                        this.td.appendChild(btUpdate);
        
                        // *** DELETE.
                        this.td = document.createElement('th');
                        tr.appendChild(this.td);
                        var btDelete = document.createElement('input');
                        btDelete.setAttribute('type', 'button');    // SET INPUT ATTRIBUTE.
                        btDelete.setAttribute('value', 'Delete');
                        btDelete.setAttribute('disabled','true') ;
                        btDelete.setAttribute('style', 'background-color:#ED5650;');
                        btDelete.setAttribute('onclick', 'Delete(this)');   // ADD THE BUTTON's 'onclick' EVENT.
                        this.td.appendChild(btDelete);
                    }

                    this.td = document.createElement('td');
                    tr.appendChild(this.td);

                    var btNew = document.createElement('input');

                    btNew.setAttribute('type', 'button');       // SET ATTRIBUTES.
                    btNew.setAttribute('value', 'Create');
                    btNew.setAttribute('id', 'New' + i);
                    btNew.setAttribute('style', 'background-color:#207DD1;');
                    btNew.setAttribute('onclick', 'CreateNew(this)');       // ADD THE BUTTON's 'onclick' EVENT.
                    this.td.appendChild(btNew);
                    ///
                    tr = table.insertRow(-1);           // CREATE THE LAST ROW.

                    for (var j = 0; j < this.col.length; j++) {
                        var newCell = tr.insertCell(-1);    
                                var tBox = document.createElement('input');          // CREATE AND ADD A TEXTBOX.
                                tBox.setAttribute('type', 'text');
                                tBox.setAttribute('id', 'Newitem' + j);
                                tBox.setAttribute('value', '');
                                newCell.appendChild(tBox);
                    } ///



                    var div = document.getElementById('container');
                    // console.log(div);
                    div.innerHTML = '';
                    div.appendChild(table);


            
                }
                

               
                

                // CREATE NEW.
                    
                    
            }
           //console.log(typeof(data.intents));
           //console.log(data.intents);
           data.intents.forEach(ele=> crudApp2.arreglo.push(ele));
           
           crudApp2.createTable();
            

    
        });
  }
    this.Cancel = function (oButton) {console.log("Cancel");}
    this.Update= function (oButton) {console.log("update");}
    this.Delete = function (oButton) {console.log("delete");}
    this.CreateNew = function (oButton) {
        console.log("create new");

        let ndata0 = document.getElementById("Newitem0");
        let ndata1 = document.getElementById("Newitem1");
        let ndata2 = document.getElementById("Newitem2");
        //console.log(ndata0.value,ndata1.value,ndata2.value);
        let njson = {
            tag: ndata0.value,
            patterns: [ndata1.value],
            responses: [ndata2.value]
        };
        let json2 =JSON.stringify(njson);
        //console.log(json2);
        
        fetch("./intents.json")
            .then((res) => {
                return res.json();
            })
            .then((data) => 
                {
                    let obj =data;
                    
                    obj['intents'].push(njson)
                    //console.log("javascript encode ",obj);

                    let response = fetch('http://127.0.0.1:5000/json', {
                        method: 'POST',
                        headers: {
                          'Content-Type': 'application/json;charset=utf-8'
                        },
                        body: JSON.stringify(obj)
                      });
                });
        }