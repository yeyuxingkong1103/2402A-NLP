// 工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。逐项目启动录制，输出不写主目录。
const {spawn}=require('child_process'),path=require('path');
const root=path.resolve(__dirname,'../..');
(async()=>{for(let i=2;i<=13;i++){const cwd=path.join(root,'工单'+i);const child=spawn(process.execPath,['测试/录制演示.cjs'],{cwd,windowsHide:true,stdio:'inherit'});await new Promise(resolve=>child.on('exit',code=>{console.log('录制工单',i,'exit',code);resolve()}))}})();
