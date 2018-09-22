require(EBImage)
require(DiagrammeR)
require(mxnet)

# USE FUNCTION DISPLAY FOR SEE AN IMAGE

# importa tutto il dataset di immagini
  #setwd("/Users/silvio_saglimbeni/Desktop/dogsVScats/train_ridotto")
  #files <- list.files(path="/Users/silvio_saglimbeni/Desktop/dogsVScats/train_ridotto",
  #                  pattern=".jpg",all.files=T, full.names=F, no.. = T)
  # ad ogni immagine applica la funzione readImage (che legge e converte ciascuna
  # immagine)
  #list_of_images = lapply(files, readImage) 

# per rendere più veloce il futuro import delle immagini le salviamo
# in formato RData per poi caricarle con il comando load
setwd("~/Desktop/dogsVScats")
  #save(list_of_images,file="list_of_images.RData")
load("list_of_images.RData") #list_of_images

# raggruppiamo tutto in un unica matrice di valori
image_matrix = do.call('cbind', lapply(list_of_images, as.numeric))


setwd("~/Desktop/dogsVScats")
# salviamo il risultato "image_matrix" in formto R
save(image_matrix,file="image_matrix.RData")

############## conversione biano e nero
 bw_list_of_image= list()
for(i in 1:length(list_of_images)){
 bw_list_of_image[[i]] = channel(list_of_images[[i]],"gray")
}

bw_image_matrix = do.call('cbind', lapply(bw_list_of_image, as.numeric))
bw_t_image_matrix= t(bw_image_matrix)
######################

#t_image_matrix= t(image_matrix)
# per ogni riga di X ti dice che soggetto rappresenta quell'immagine
# 0 = cat
# 1 = dog
labels = rep(1,dim(image_matrix)[2])
n_cat=round( (dim(image_matrix)[2]/2) ,0)
labels[1:n_cat]=0
# Dataframe of resized images
rs_df <- data.frame()

# Main loop: for each image, resize and set it to greyscale
for(i in 1:nrow(t_image_matrix))
{
  # Try-catch
  result <- tryCatch({
    #' Image (as 1d vector), Ogni riga di X è un immagine 64 x 64 pixel
    #' che è quindi una riga 64 x 64 = 4096, quindi una riga con 
    #'  4096 pixel (è quindi una riga con 4096 valori)
    #'  
    img <- as.numeric(bw_t_image_matrix[i,])
    
    # Reshape as a 64x64 image (EBImage object), riplasma l'immagine
    # come oggetto Ebi, ovvero ogni riga diventa una matrice 64 x 64
    # dove ogni valore rappresenta quel pixel
    # > dim(img@.Data)
    # [1] 64 64
    dimensione=list_of_images[[i]]@dim
    dimensione = dimensione[-3]
    img <- Image(img,dim= dimensione ,colormode = "Grayscale")
    # Resize image to 28x28 pixels (Rimpicciolisci l'immagine per ridurre
    # la complessita computazionale, in una 28x28)
    # > dim(img_resized@.Data)
    # [1] 28 28
    img_resized <- resize(img, w = 28, h = 28)
    # Get image matrix (there should be another function 
    # to do this faster and more neatly!) 
    # L'oggetto img era un oggetto con tante cose, a noi interessa solo la
    # matrice con i valori dei pixel
    # > dim(img_matrix)
    # [1] 28 28
    img_matrix <- img_resized@.Data
    # Coerce to a vector, ritrasforma tale matrice in un vettore (riga)
    img_vector <- as.vector(t(img_matrix))
    # Add label, a questo vettore aggiungiamo come primo valore 
    # l'immagine a cui corrisponde, per esempio se tale riga corrisponde ad
    # una foto del tizio 3 il primo valore di tale vettore sarà 3
    label <- labels[i]
    vec <- c(label, img_vector)
    # Stack in rs_df using rbind, tale vettore viene aggiunto come riga 
    # del dataframe rs_df
    rs_df <- rbind(rs_df, vec)
    # Print status
    print(paste("Done",i,sep = " "))},
    # Error function (just prints the error). Btw you should get no errors!
    error = function(e){print(e)})
}
#' in pratica con questo ciclo abbiamo creato un dataset che ha 
#' come primo valore la codifica dell'indivioduo in fotografia 
#' e come rimanenti 28 x 28 colonne i valori dei pixel tra 0 e 1
# vengono ora denominate le colonne come tali
# Set names. The first columns are the labels, the other columns are the pixels.
# names(rs_df) <- c("label", paste("pixel", c(1:784)))


# Shuffled df
shuffled <- rs_df[sample(1:NROW(rs_df)),]
ntrain=round(NROW(rs_df)*0.7,0)
# Train-test split
train_28 <- shuffled[1:ntrain, ]
test_28 <- shuffled[ntrain:NROW(rs_df), ]
# Set up train and test datasets
# trasformiamo in matrici e ridividiamo tra x e y
train <- data.matrix(train_28)
train_x <- t(train_28[, -1])
train_y <- train_28[, 1]
train_array <- train_x
dim(train_array) <- c(28, 28, 1, ncol(train_x))
#' serve un arrey perchè ogni riga è in realta una matrice (una
#' foto con 28 x 28 valori), quindi 40 matrici assieme sono un 
#' oggetto array

test_x <- t(test_28[, -1])
# ora abbiamo i pixel in riga e le immagini in colonna
test_y <- test_28[, 1]
# lo stesso per i relativi individui in foto
test_array <- test_x
# con tale logica ridefiniamo le dimensioni dell'array
dim(test_array) <- c(28, 28, 1, ncol(test_x))




data <- mx.symbol.Variable('data')
# 1st convolutional layer
conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 2nd convolutional layer
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 1st fully connected layer
flatten <- mx.symbol.Flatten(data = pool_2)
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "tanh")
# 2nd fully connected layer
fc_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 40)
# Output. Softmax output since we'd like to get some probabilities.
NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)

# Device used. CPU in my case.
devices <- mx.cpu()

# Train the model
model <- mx.model.FeedForward.create(NN_model,
                                     X = train_array,
                                     y = train_y,
                                     ctx = devices,
                                     num.round = 480,
                                     array.batch.size = 40,
                                     learning.rate = 0.01,
                                     momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

# Testing
# Predict labels
predicted <- predict(model, test_array)
# Assign labels
predicted_labels <- max.col(t(predicted)) - 1

print("confusion Matrix")
table(test_28[, 1], predicted_labels)
print("Get accuracy")
sum(diag(table(test_28[, 1], predicted_labels)))/ sum(table(test_28[, 1], predicted_labels))