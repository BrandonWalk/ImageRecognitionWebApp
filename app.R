library(shiny)
library(keras)
library(jpeg)
library(stringr)

ui <- fluidPage(

    titlePanel("Brandon's Image Recognition Web App"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
           textInput("url", "Choose an image from the web that is a .jpg file:", value = 'https://upload.wikimedia.org/wikipedia/commons/1/1e/Bald_Eagle_Portrait.jpg')
        ),
        mainPanel(
          h3("Please give some initial loading time"),
          imageOutput("imageA", height = "400px", width = "400px"),
          verbatimTextOutput("guess")
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output, session){

  model <- application_resnet50(weights = "imagenet")

  output$guess <- reactive({
    download.file(input$url, 'data/download.jpg', mode = 'wb')
    im <- image_load("data/download.jpg", target_size = c(224,224))
    x <- image_to_array(im)
    x <- array_reshape(x, c(1, dim(x)))
    x <- imagenet_preprocess_input(x)
    preds <- predict(model, x)
    str_to_title(str_replace(imagenet_decode_predictions(preds, top = 3)[[1]]$class_description[1], "_", " "))
  })

  output$imageA <- renderImage({
    download.file(input$url, 'data/download.jpg', mode = 'wb')
    list(
      src = "data/download.jpg",
      contentType = "image/jpeg",
      height = 390,
      alt = "image you selected"
    )
  })

}

# Run the application 
shinyApp(ui = ui, server = server)
