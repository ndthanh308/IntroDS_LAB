// java old

private State newState() {
    return new State(Collections. < Entry > emptyList(), getBaseState());
}

// java new
private ConfigSnapshot newState() {
    return new ConfigSnapshot(Collections. < ConfigLine > emptyList(), getBaseState());
}

// c# old
private Config.State NewState() {
    return new Config.State(Sharpen.Collections.EmptyList < Config.Entry > (), GetBaseState());
}

// c# new
private ConfigSnapshot NewState() {
    return new ConfigSnapshot(Sharpen.Collections.EmptyList < ConfigLine > (), GetBaseState());
}