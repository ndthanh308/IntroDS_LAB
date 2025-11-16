settings.outformat = "pdf";
int std_size = 170;

int n = 4;
int M;
int r = 4;

int[][] K_to_A(pair[] K) {
    int[][] A;
    for (pair k : K) {
        int[] a;
        for (int i = ceil(k.x)-1; i <= floor(k.y); ++i) {
            a.push(i);
        }
        A.push(a);
    }
    return A;
}

pair[][] A_to_B(int[][] A, int[] d) {
    M = max(A);
    for (int[] a : A) {
        M = max(M, max(a));
    }
    pair[][] B;
    for (int i = 0; i < A.length; ++i) {
        pair[] b;
        for (int a : A[i]) {
            b.push((a, M+2*(d[i]-1)-a));
            b.push((a, M+2*(d[i]-1)+1-a));
        }
        B.push(b);
    }
    return B;
}

int[][] B_to_eta(pair[][] B, int[] d) {
    int s = M+2*(r-1) + 2;
    int[][] eta = array(s, array(s, r));
    for (int i = 0; i < s; ++i) {
        for (int j = 0; j < s; ++j) {
            if (i+j >= M+2*(r-1)) {
                eta[i][j] = 0;
            }
        }
    }
    for (int i = 0; i < B.length; ++i) {
        for (pair b : B[i]) {
            eta[floor(b.x)][floor(b.y)] = r-d[i];
        }
    }
    return eta;
}

int[][] eta_to_chi(int[][] eta) {
    int[][] chi = array(eta.length, array(eta[0].length, 0));
    for (int i = 0; i < chi.length; ++i) {
        for (int j = 0; j < chi[0].length; ++j) {
            chi[i][j] = eta[i][j];
            if (i > 0) chi[i][j] = min(chi[i][j], chi[i-1][j]);
            if (j > 0) chi[i][j] = min(chi[i][j], chi[i][j-1]);
        }
    }
    return chi;
}

void draw_K(pair[] K, int[] d, pen[] pens, pair[] Klpos, string[] labels) {
    picture pic_K;
    size(std_size,0,pic=pic_K);
    for (int i = 0; i < K.length; ++i) {
        draw((K[i].x,d[i]*1.5) -- (K[i].y,d[i]*1.5), pic=pic_K, p=(pens[i]+1.5));
        label("$K_"+labels[i]+"$", Klpos[i], p=pens[i], pic=pic_K);
    }
    shipout("pic_K", pic=pic_K);
}

void draw_A(int[][] A, int[] d, pen[] pens, pair[] Alpos, string[] labels) {
    picture pic_A;
    size(std_size,0,pic=pic_A);
    for (int i = 0; i < A.length; ++i) {
        for (int a : A[i]) {
            dot((a,d[i]*1.5), pic=pic_A, p=pens[i]);
        }
        label("$A_"+labels[i]+"$", Alpos[i], p=pens[i], pic=pic_A);
    }
    shipout("pic_A", pic=pic_A);
}

void draw_B(pair[][] B, pen[] pens, pair[] Blpos, string[] labels) {
    picture pic_B;
    size(std_size,0,pic=pic_B);
    for (int i = 0; i < B.length; ++i) {
        for (pair b : B[i]) {
            dot(b, pic=pic_B, p=pens[i]);
        }
        label("$B_"+labels[i]+"$", Blpos[i], p=pens[i], pic=pic_B);
    }
    shipout("pic_B", pic=pic_B);
}

void draw_table(int[][] T, string name) {
    defaultpen(fontsize(5pt));
    picture pic_T;
    size(std_size,0,pic=pic_T);
    for (int x = 0; x < T.length; ++x) {
        for (int y = 0; y < T[x].length; ++y) {
            label(string(T[x][y]), (x, y), pic=pic_T);
            if (x == 0 || T[x][y] != T[x-1][y]) {
                draw((x-0.5,y-0.5)--(x-0.5,y+0.5), pic=pic_T);
            }
            if (y == 0 || T[x][y] != T[x][y-1]) {
                draw((x-0.5,y-0.5)--(x+0.5,y-0.5), pic=pic_T);
            }
        }
    }
    shipout(name, pic=pic_T);
}


pen[] pcols = {red, heavygreen, blue, orange};
pcols.cyclic = true;
pen[] pshapes = {makepen(scale(1.5)*unitcircle), makepen(scale(2)*polygon(4)), makepen(scale(2)*polygon(3)),
    makepen(scale(1.2)*((0.4,0)--(1.2,0.8)--(0.8,1.2)--(0,0.4)--(-0.8,1.2)--(-1.2,0.8)--(-0.4,0)--(-1.2,-0.8)--(-0.8,-1.2)--(0,-0.4)--(0.8,-1.2)--(1.2,-0.8)--cycle))};
pshapes.cyclic = true;
pen[] pens = sequence(new pen(int i) {return pcols[i]+pshapes[i];}, pcols.length);
pens.cyclic = true;
int[] d = {1, 2, 1, 3};
pair[] K = {(0.3, 2.2),  (1.7, 18.1), (15.7, 18.9), (16.2, 21.6)};

int[][] A = K_to_A(K);
pair[][] B = A_to_B(A,d);
int[][] eta = B_to_eta(B,d);
int[][] chi = eta_to_chi(eta);


pair[] Klpos = {(1,0),(9.7,4.5),(17.5,0),(19.7,6)};
pair[] Alpos = {(1,0),(9.7,4.5),(16.5,0),(19,6)};
pair[] Blpos = {(0,19),(9.7,16.3),(15.5,3.5),(19.7,8.3)};
string[] labels = {"1","3","2","4"};

draw_K(K, d, pcols, Klpos, labels);
draw_A(A, d, pens, Alpos, labels);
draw_B(B, pens, Blpos, labels);
draw_table(eta, "pic_eta");
draw_table(chi, "pic_chi");
