{
//TCanvas *c1 = new TCanvas("c1","bplots",1800,1200);
//c1->SetFillColor(10);
gROOT->ProcessLine(".x /home/natasa/analiza/goran/CLICdpStyle.C");
//gStyle->SetOptStat(1111);	
gStyle->SetOptFit(1);	



//TF1 *f1 = new TF1("f1", "[0] +  [1] * (sin(x * (0.5 + [2]/3.14) ))^2", -3.1, 3.1);
TF1 *f1 = new TF1("f1", "[0] +  [1] * (sin(x * (0.5 + [2]/3.14) ))^2 + [3] * (cos(x * (0.5 + [2]/3.14) ))^2 ", -3.1, 3.1);

f1 -> SetParameters(14.,15.,0., 1.);
//f1 -> SetParameters(14.,15.,0.);
f1 -> SetParLimits(2, 0., 1.6);

TFile *f31 = new TFile ("ugao_mc_ilc_scalar_full_range.root", "read");//full range whizard
TFile *f32 = new TFile ("ugao_mc_ilc_scalar_trk.root", "read");//ugao tracker
TFile *g = new TFile ("signal_whole.root", "read");//full range signal


TTree * tr = (TTree *)f31->Get("events");
TTree * tr1 = (TTree *)f32->Get("events");

TTree * trg = (TTree *)g->Get("leptonTree");


TH1F *hgen = new TH1F("hgen", " ",100,-3.1, 3.1); // whizard
tr->Draw("fi>>hgen", "abs(fi)<3.1");
hgen->SetLineColor(1);
hgen->GetXaxis()->SetTitle("#Delta#Phi [rad]");
hgen->GetYaxis()->SetTitle("count/1 ab^{-1}");
hgen->GetYaxis()->SetTitleOffset(0.7);
hgen->Scale(2550./hgen->Integral());

TH1F *h11 = new TH1F("h11", "",100,-3.1,3.1);
tr->Draw("fi>>h11"); 
h11 ->Scale(2550./h11->Integral());



TH1F *h22 = new TH1F("h22", "",100,-3.1,3.1); // rekontruisani  signal
tr1->Draw("fi>>h22", "abs(fi)<3.1  ");
h22 ->Scale(2550./h22->Integral()); 


h11->Divide(h22);
double odbroj = h11->Integral();
h11->Scale(100/odbroj);
h11->Draw();// correction function za acceptance


//////////////////////////////////////////////
TH1F *hfi = new TH1F("hfi", "",100,-3.1,3.1);
TH1F *ht = new TH1F("ht", "",100,-3.1,3.1);
TH1F *hs = new TH1F("hs", "",100,-3.1,3.1); // rekontruisani  signal

trg->Draw("fi>>hs", "m_H > 100 && m_H<160 && m_Z2<500 && pt_e_sistema > 30 && logy_12 < 3 && m_Z1>21 && pt_miss<150 ");
hs ->Scale(2550./hs->Integral()); 
trg->Draw("fi>>ht", "m_H > 100 && m_H<160 && m_Z2<500 && pt_e_sistema > 30 && logy_12 < 3 && m_Z1>21 && pt_miss<150 ");
ht ->Scale(2550./ht->Integral());
double sc = 2550./ht->Integral();

for (int i = 0; i < 100; i++) {
//  cout << i << "\n";
  float p = h11 -> GetBinContent(i);
//cout <<"sadrzaj "<< p << endl;
float q = hs -> GetBinContent(i);

double err;
err = ht->GetBinError(i)*sc;

hfi->SetBinContent(i, p*q);
hfi->SetBinError(i, err);

}

hfi->GetXaxis()->SetTitle("#Delta#Phi [rad]");
hfi->GetYaxis()->SetTitle("count/1 ab^{-1}");


hfi->Scale(2550/hfi->Integral());
hfi ->Draw("hist");



//hfi -> Fit(f1, "", "", -3.1, 3.1);
//hgen -> Fit(f1, "", "", -3.1, 3.1);
//f1->Draw("same");
//hgen->Draw("hist same");

//legend->AddEntry("hfi","selected corrected","l");
//legend->AddEntry("hgen","generated full range","l");
//legend->SetTextSize(.05);
//legend->Draw();
//h->Add(hgen);
/*
c1->cd(2);
hgen->GetXaxis()->SetTitle("#Delta#Phi [rad]");
hgen->GetYaxis()->SetTitle("count/2.5 ab^{-1}");
hgen->GetYaxis()->SetTitleOffset(0.7);
hgen -> Fit(f1, "", "", -3.1, 3.1);


hsel->Draw("hist");
hfi->Draw("hist same");
hgen->Draw("hist same");


c1->cd(3);
hfi->GetYaxis()->SetTitleSize(0.06);
hfi->GetYaxis()->SetTitleOffset(0.9);
hfi -> Fit(f1, "", "", -3.1, 3.1);
//hsel -> Fit(f1, "", "", -3.1, 3.1);
float err = f1->GetParError(2);
//float br = f2->GetParameter(2);

psi_err = err;*/// * 1000
//psi_mean_DJ = br;


}
