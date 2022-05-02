# frozen_string_literal: true

Rails.application.routes.draw do
  root 'fours#index'

  resource :emp, only: [:create]
  resource :emp_three, only: %i[show create]
  resource :four, only: %i[create]
end
