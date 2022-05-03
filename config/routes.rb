# frozen_string_literal: true

Rails.application.routes.draw do
  root 'corellations#index'

  resource :emp, only: [:create]
  resource :emp_three, only: %i[show create]
  resource :corellation, only: %i[create]
end
