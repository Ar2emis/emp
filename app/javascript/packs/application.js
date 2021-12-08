import jquery from "jquery"
import Rails from "@rails/ujs"
import Turbolinks from "turbolinks"
import * as ActiveStorage from "@rails/activestorage"
import "chartkick/chart.js"


Rails.start()
Turbolinks.start()
ActiveStorage.start()

window.$ = window.jquery = jquery;
