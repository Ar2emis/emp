# frozen_string_literal: true

Rails.application.routes.draw do
  root 'fives#index'

  resource :emp, only: [:create]
  resource :emp_three, only: %i[show create]
  resource :four, only: %i[create]
  resource :five, only: %i[create]
end
