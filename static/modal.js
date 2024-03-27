$(function () {
    // ボタンをクリックしたらモーダル表示
  $(".modalBtn").on("click", function () {
    $(".modalBg").fadeIn();
    return false;
  });
  // ×ボタンクリックでモーダル閉じる
  $(".modalClose").on("click", function () {
    $(".modalBg").fadeOut();
  });
  // モーダルコンテンツ以外をクリックでモーダル閉じる
  $(document).on("click", function (e) {
    if (!$(e.target).closest(".modalArea").length) {
      $(".modalBg").fadeOut();
    }
  });
});