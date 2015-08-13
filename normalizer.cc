/// Perform 1D Gaussian convolutions using a FIR filter.
///
/// The mask is computed to 3 sigma.

template<class T>
void gauss1d(mdarray<T> &out, mdarray<T> &in, float sigma) {
    out.resize(in.dim(0));
    // make a normalized mask
    int range = 1+int(3.0*sigma);
    floatarray mask(2*range+1);
    for (int i = 0; i <= range; i++) {
        double y = exp(-i*i/2.0/sigma/sigma);
        mask(range+i) = mask(range-i) = y;
    }
    float total = 0.0;
    for (int i = 0; i < mask.dim(0); i++)
        total += mask(i);
    for (int i = 0; i < mask.dim(0); i++)
        mask(i) /= total;

    // apply it
    int n = in.size();
    for (int i = 0; i < n; i++) {
        double total = 0.0;
        for (int j = 0; j < mask.dim(0); j++) {
            int index = i+j-range;
            if (index < 0)
                index = 0;
            if (index >= n)
                index = n-1;
            total += in(index) * mask(j);  // it's symmetric
        }
        out(i) = T(total);
    }
}

template void gauss1d(bytearray &out, bytearray &in, float sigma);
template void gauss1d(floatarray &out, floatarray &in, float sigma);

/// Perform 1D Gaussian convolutions using a FIR filter.
///
/// The mask is computed to 3 sigma.

template<class T>
void gauss1d(mdarray<T> &v, float sigma) {
    mdarray<T> temp;
    gauss1d(temp, v, sigma);
    v.take(temp);
}

template void gauss1d(bytearray &v, float sigma);
template void gauss1d(floatarray &v, float sigma);

/// Perform 2D Gaussian convolutions using a FIR filter.
///
/// The mask is computed to 3 sigma.

template<class T>
void gauss2d(mdarray<T> &a, float sx, float sy) {
    floatarray r, s;
    for (int i = 0; i < a.dim(0); i++) {
        getd0(a, r, i);
        gauss1d(s, r, sy);
        putd0(a, s, i);
    }
    for (int j = 0; j < a.dim(1); j++) {
        getd1(a, r, j);
        gauss1d(s, r, sx);
        putd1(a, s, j);
    }
}

template void gauss2d(bytearray &image, float sx, float sy);
template void gauss2d(floatarray &image, float sx, float sy);

void argmax1(mdarray<float> &m, mdarray<float> &a) {
    m.resize(a.dim(0));
    for (int i = 0; i < a.dim(0); i++) {
        float mv = a(i, 0);
        float mj = 0;
        for (int j = 1; j < a.dim(1); j++) {
            if (a(i, j) < mv) continue;
            mv = a(i, j);
            mj = j;
        }
        m(i) = mj;
    }
}

struct CenterNormalizer : INormalizer {
    pymulti::PyServer *py = 0;
    mdarray<float> center;
    float r = -1;
    void setPyServer(void *p) {
        this->py = (pymulti::PyServer*)p;
    }
    void getparams(bool verbose) {
        range = getrenv("norm_range", 4.0);
        smooth2d = getrenv("norm_smooth2d", 1.0);
        smooth1d = getrenv("norm_smooth1d", 0.3);
        if (verbose) print("center_normalizer", range, smooth2d, smooth1d);
    }
    void measure(mdarray<float> &line) {
        mdarray<float> smooth, smooth2;
        int w = line.dim(0);
        int h = line.dim(1);
        smooth.copy(line);
        gauss2d(smooth, h*smooth2d, h*0.5);
        add_smear(smooth, line);  // just to avoid singularities
        mdarray<float> a(w);
        argmax1(a, smooth);
        gauss1d(center, a, h*smooth1d);
        float s1 = 0.0;
        float sy = 0.0;
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                s1 += line(i, j);
                sy += line(i, j) * fabs(j-center(i));
            }
        }
        float mad = sy/s1;
        r = int(range*mad+1);/// Perform 1D Gaussian convolutions using a FIR filter.
///
/// The mask is computed to 3 sigma.

template<class T>
void gauss1d(mdarray<T> &out, mdarray<T> &in, float sigma) {
    out.resize(in.dim(0));
    // make a normalized mask
    int range = 1+int(3.0*sigma);
    floatarray mask(2*range+1);
    for (int i = 0; i <= range; i++) {
        double y = exp(-i*i/2.0/sigma/sigma);
        mask(range+i) = mask(range-i) = y;
    }
    float total = 0.0;
    for (int i = 0; i < mask.dim(0); i++)
        total += mask(i);
    for (int i = 0; i < mask.dim(0); i++)
        mask(i) /= total;

    // apply it
    int n = in.size();
    for (int i = 0; i < n; i++) {
        double total = 0.0;
        for (int j = 0; j < mask.dim(0); j++) {
            int index = i+j-range;
            if (index < 0)
                index = 0;
            if (index >= n)
                index = n-1;
            total += in(index) * mask(j);  // it's symmetric
        }
        out(i) = T(total);
    }
}

template void gauss1d(bytearray &out, bytearray &in, float sigma);
template void gauss1d(floatarray &out, floatarray &in, float sigma);

/// Perform 1D Gaussian convolutions using a FIR filter.
///
/// The mask is computed to 3 sigma.

template<class T>
void gauss1d(mdarray<T> &v, float sigma) {
    mdarray<T> temp;
    gauss1d(temp, v, sigma);
    v.take(temp);
}

template void gauss1d(bytearray &v, float sigma);
template void gauss1d(floatarray &v, float sigma);

/// Perform 2D Gaussian convolutions using a FIR filter.
///
/// The mask is computed to 3 sigma.

template<class T>
void gauss2d(mdarray<T> &a, float sx, float sy) {
    floatarray r, s;
    for (int i = 0; i < a.dim(0); i++) {
        getd0(a, r, i);
        gauss1d(s, r, sy);
        putd0(a, s, i);
    }
    for (int j = 0; j < a.dim(1); j++) {
        getd1(a, r, j);
        gauss1d(s, r, sx);
        putd1(a, s, j);
    }
}

template void gauss2d(bytearray &image, float sx, float sy);
template void gauss2d(floatarray &image, float sx, float sy);

void argmax1(mdarray<float> &m, mdarray<float> &a) {
    m.resize(a.dim(0));
    for (int i = 0; i < a.dim(0); i++) {
        float mv = a(i, 0);
        float mj = 0;
        for (int j = 1; j < a.dim(1); j++) {
            if (a(i, j) < mv) continue;
            mv = a(i, j);
            mj = j;
        }
        m(i) = mj;
    }
}

struct CenterNormalizer : INormalizer {
    pymulti::PyServer *py = 0;
    mdarray<float> center;
    float r = -1;
    void setPyServer(void *p) {
        this->py = (pymulti::PyServer*)p;
    }
    void getparams(bool verbose) {
        range = getrenv("norm_range", 4.0);
        smooth2d = getrenv("norm_smooth2d", 1.0);
        smooth1d = getrenv("norm_smooth1d", 0.3);
        if (verbose) print("center_normalizer", range, smooth2d, smooth1d);
    }
    void measure(mdarray<float> &line) {
        mdarray<float> smooth, smooth2;
        int w = line.dim(0);
        int h = line.dim(1);
        smooth.copy(line);
        gauss2d(smooth, h*smooth2d, h*0.5);
        add_smear(smooth, line);  // just to avoid singularities
        mdarray<float> a(w);
        argmax1(a, smooth);
        gauss1d(center, a, h*smooth1d);
        float s1 = 0.0;
        float sy = 0.0;
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                s1 += line(i, j);
                sy += line(i, j) * fabs(j-center(i));
            }
        }
        float mad = sy/s1;
        r = int(range*mad+1);
        if (py) {
            print("r", r);
            py->eval("ion(); clf()");
            py->eval("subplot(211)");
            py->imshowT(line, "cmap=cm.gray,interpolation='nearest'");
            py->eval("subplot(212)");
            py->imshowT(smooth, "cmap=cm.gray,interpolation='nearest'");
            py->plot(center);
            py->eval("print ginput(999)");
        }
    }
    void normalize(mdarray<float> &out, mdarray<float> &in) {
        int w = in.dim(0);
        if (w != center.dim(0)) THROW("measure doesn't match normalize");
        float scale = (2.0 * r) / target_height;
        int target_width = max(int(w/scale), 1);
        out.resize(target_width, target_height);
        for (int i = 0; i < out.dim(0); i++) {
            for (int j = 0; j < out.dim(1); j++) {
                float x = scale * i;
                float y = scale * (j-target_height/2) + center(int(x));
                out(i, j) = bilin(in, x, y);
            }
        }
    }
};
        if (py) {
            print("r", r);
            py->eval("ion(); clf()");
            py->eval("subplot(211)");
            py->imshowT(line, "cmap=cm.gray,interpolation='nearest'");
            py->eval("subplot(212)");
            py->imshowT(smooth, "cmap=cm.gray,interpolation='nearest'");
            py->plot(center);
            py->eval("print ginput(999)");
        }
    }
    void normalize(mdarray<float> &out, mdarray<float> &in) {
        int w = in.dim(0);
        if (w != center.dim(0)) THROW("measure doesn't match normalize");
        float scale = (2.0 * r) / target_height;
        int target_width = max(int(w/scale), 1);
        out.resize(target_width, target_height);
        for (int i = 0; i < out.dim(0); i++) {
            for (int j = 0; j < out.dim(1); j++) {
                float x = scale * i;
                float y = scale * (j-target_height/2) + center(int(x));
                out(i, j) = bilin(in, x, y);
            }
        }
    }
};