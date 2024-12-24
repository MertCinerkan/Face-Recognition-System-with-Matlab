clear
camera = webcam(); % Kamerayı bağla
load('trainedModelSVM.mat'); % Eğitilmiş modeli yükle
dete = vision.CascadeObjectDetector();

while true   
    picture = camera.snapshot; % Bir fotoğraf çek    
   
    % RGB görüntüyü griye dönüştür ve LBP özelliklerini çıkar
    net = alexnet;
    inputsize = net.Layers(1).InputSize;
    aug = augmentedImageDatastore(inputsize(1:2), picture);
    layer = 'fc8';
    features = activations(net, aug, layer, 'OutputAs', 'rows');
    
    % Yüzleri tespit et
    bbox = detect(dete, picture);
    
    % Yüzleri kare içine al
    detpic = insertObjectAnnotation(picture, 'rectangle', bbox, 'Face');
    
    % Eğitilmiş modeli kullanarak yüz tahmini yap  
    label = predict(trainedModelSVM.ClassificationSVM, features);
    
    % Görüntüyü göster
    image(detpic);
    title(label)
end