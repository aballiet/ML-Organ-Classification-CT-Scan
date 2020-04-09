var exec = require('child_process').exec, child;
var fs = require('fs');

var files = fs.readdirSync('./segmentations/');

var patient = ["10000100", "10000104", "10000105", "10000106", "10000108", "10000109", "10000110", "10000111", 
"10000112", "10000113", "10000127", "10000128", "10000129", "10000130", "10000131", "10000132", "10000133", "10000134", "10000135", "10000136"]

patient.forEach(element => {
    console.log("ExtractBlocks.exe -i volumes/"+ element+"_1_CTce_ThAb.nii.gz -s "+getList(element)+"-n 1000")

})


function getList(txt){
    let result = ""
    let count = 0;
    files.forEach(element => {
        if (element.includes(txt)){
            result += "segmentations/"+element+" ";
            count++
        }
    });

    console.log("il y a : " + count + " elements")

    return result;
}


function runCommand(txt){
    exec(txt,
    function (error, stdout, stderr) {
        console.log('stdout: ' + stdout);
        console.log('stderr: ' + stderr);
        if (error !== null) {
             console.log('exec error: ' + error);
        }
    });
} 
