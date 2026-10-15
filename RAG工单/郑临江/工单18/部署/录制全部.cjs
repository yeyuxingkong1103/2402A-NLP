// 工单14-18：按各项目独立脚本录制。
const {spawn}=require('child_process'),path=require('path');const root=path.resolve(__dirname,'../..');
(async()=>{for(let n=14;n<=18;n++){const child=spawn(process.execPath,['测试/录制项目.cjs'],{cwd:path.join(root,'工单'+n),windowsHide:true,stdio:'inherit'});await new Promise(resolve=>child.on('exit',code=>{console.log('video',n,'exit',code);resolve()}))}})();
