document.getElementById('loading_Form').onsubmit = function(event) {
    // デフォルトのフォーム送信を防止
    event.preventDefault();

    // ローディング画面を表示
    document.getElementById('loading').style.display = 'block';

    // FormDataオブジェクトを作成
    var formData = new FormData(this);

    // fetch APIを使用して非同期でデータを送信
    fetch(this.action, {
        method: 'POST',
        body: formData,
    })
    .then(response => {
        if(response.ok) {
            return response.json(); // または、次のページへリダイレクト
        }
        throw new Error('Network response was not ok.');
    })
    .then(data => {
        // 処理が成功した後の処理。例えば、成功メッセージを表示するか、リダイレクトする。
        window.location.href = '/next_page'; // 次のページへリダイレクト
    })
    .catch(error => {
        console.error('There has been a problem with your fetch operation:', error);
    });
};