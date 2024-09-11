const jsonString = `{
  "address":"Busan",
  "country":"Korea",
  "detail":{
    "address":"Hadan"
  }
}`;

const jsonObj = JSON.parse(jsonString);
console.log(jsonObj.detail);