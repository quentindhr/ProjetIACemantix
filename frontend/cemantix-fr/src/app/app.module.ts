import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClient, HttpClientModule } from '@angular/common/http';

// Note: AppComponent is a standalone component (see app.component.ts) and is bootstrapped directly.
import { AppComponent } from './app.component';
import { CommonModule } from '@angular/common';

@NgModule({
  // pas de declarations car AppComponent est standalone
  imports: [
    BrowserModule,
    HttpClient,
    CommonModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }