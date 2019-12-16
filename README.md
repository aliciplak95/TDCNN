# TDCNN
## Termal Kameralardan Alınan Görüntülerin Gürültülerden Arındırılması
Bu proje [hezw2016](https://github.com/hezw2016/DLS-NUC)'dan alıntılanarak geliştirilmiştir.

Projede termal kamerada oluşan dikey gürültü çizgilerinin yapay sinir ağları oluşturularak derin öğrenme ile belirli parametreler verilerek temizlenmesi amaçlanmaktadır. Çalışmamızda aşağıdaki blog diyagramında da gösterilen akış diyagramında oluşturulan katmanlar kullanılarak eğitim modeli oluşturulmuştur. Model gürültülü giriş görüntüsünü konvolüsyon, relu ve 15 adet konvolüsyon, batch normalization, relu işleminden geçirerek oluşan dikey gürültü çizgilerini giriş görüntüsünden çıkartarak gürültüyü temizlemeyi sağlar. Bu proje sağlık alanlarında hastalıkların teşhisi için çekilen EKG ya da tomografi vb. görüntülerin netleştirilmesin de ya da mobese, güvenlik kameraları gibi suçlu tespiti yapmak için çekilen görüntülerin gürültüden arındırılmasında kullanılabilir.

![blok_şema](https://github.com/aliciplak95/TDCNN/blob/master/results/tdcnn.png)

Model geliştirme sürecinde FLIR datasetinden 8000 termal kamera görüntüsü kullanılmıştır.
Test edilen verilerde PSNR(Gürültü temizleme oranı) = 25 dB olarak kaydedilmiştir.
